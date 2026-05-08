//===----------------- ModulesBuilder.cpp ------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ModulesBuilder.h"
#include "Compiler.h"
#include "SourceCode.h"
#include "support/Logger.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ModuleCache.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include <chrono>
#include <ctime>

namespace clang {
namespace clangd {

namespace {

llvm::cl::opt<bool> DebugModulesBuilder(
    "debug-modules-builder",
    llvm::cl::desc("Don't remove clangd's built module files for debugging. "
                   "Remember to remove them later after debugging."),
    llvm::cl::init(false));

llvm::cl::opt<unsigned> VersionedModuleFileGCThresholdSeconds(
    "modules-builder-versioned-gc-threshold-seconds",
    llvm::cl::desc("Delete versioned copy-on-read module files whose last "
                   "access time is older than this many seconds."),
    llvm::cl::init(3 * 24 * 60 * 60));

//===----------------------------------------------------------------------===//
// Persistent Module Cache Layout.
//
// clangd publishes prerequisite BMIs into a stable on-disk cache so later
// builders can reuse them across sessions. Cache entries are grouped by a
// readable module-unit source directory name plus a hash of the normalized
// source path, and are further separated by a hash of the full compile
// command, which keeps incompatible BMI variants apart.
//
//   module-unit source
//      |
//      v
//   cache root
//      |
//      +-- <module-unit-source-name>-<source-hash>
//             |
//             +-- <command-hash>
//                    |
//                    +-- <primary-module>[-<partition>].pcm
//===----------------------------------------------------------------------===//

std::string hashStringForCache(llvm::StringRef Content) {
  return llvm::toHex(digest(Content));
}

std::string normalizePathForCache(PathRef Path) {
  llvm::SmallString<256> Normalized(Path);
  llvm::sys::path::remove_dots(Normalized, /*remove_dot_dot=*/true);
  return maybeCaseFoldPath(Normalized);
}

/// Returns the root directory used for persistent module cache storage.
/// Prefer a project-local cache so different clangd sessions working on the
/// same source tree can reuse BMIs. Fall back to the user cache directory, and
/// finally to a non-ephemeral temp directory when no better cache root exists.
llvm::SmallString<256>
getModuleCacheRoot(PathRef ModuleUnitFileName,
                   const GlobalCompilationDatabase &CDB) {
  llvm::SmallString<256> Result;
  if (auto PI = CDB.getProjectInfo(ModuleUnitFileName);
      PI && !PI->SourceRoot.empty()) {
    Result = PI->SourceRoot;
    llvm::sys::path::append(Result, ".cache", "clangd", "modules");
    return Result;
  }

  if (llvm::sys::path::cache_directory(Result)) {
    llvm::sys::path::append(Result, "clangd", "modules");
    return Result;
  }

  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/false, Result);
  llvm::sys::path::append(Result, "clangd", "modules");
  return Result;
}

/// Returns the directory holding source-scoped lock files for the persistent
/// module cache. Placing locks beside the cache ensures all builders sharing
/// the cache also synchronize through the same lock namespace.
llvm::SmallString<256>
getModuleCacheLocksDirectory(PathRef ModuleUnitFileName,
                             const GlobalCompilationDatabase &CDB) {
  llvm::SmallString<256> Result = getModuleCacheRoot(ModuleUnitFileName, CDB);
  llvm::sys::path::append(Result, ".locks");
  return Result;
}

std::string getModuleUnitSourcePathHash(PathRef ModuleUnitFileName) {
  return hashStringForCache(normalizePathForCache(ModuleUnitFileName));
}

std::string getModuleUnitSourceDirectoryName(PathRef ModuleUnitFileName) {
  std::string Result = llvm::sys::path::filename(ModuleUnitFileName).str();
  Result.push_back('-');
  Result.append(getModuleUnitSourcePathHash(ModuleUnitFileName));
  return Result;
}

std::string getCompileCommandStringHash(const tooling::CompileCommand &Cmd) {
  std::string SerializedCommand;
  SerializedCommand.reserve(Cmd.Directory.size() + Cmd.Filename.size() +
                            Cmd.CommandLine.size() * 16);
  // The module-unit source path is already encoded in the parent cache
  // directory. Output is rewritten while staging the BMI, so hash only the
  // semantic compile command to keep the cache key stable across rebuilds.
  SerializedCommand.append(Cmd.Directory);
  SerializedCommand.push_back('\0');
  for (const auto &Arg : Cmd.CommandLine) {
    SerializedCommand.append(Arg);
    SerializedCommand.push_back('\0');
  }
  return hashStringForCache(SerializedCommand);
}

/// Returns the directory for a persistent BMI built from a specific module
/// unit source and compile command. The directory name keeps a readable source
/// basename alongside the source-hash, and the command-hash keeps incompatible
/// command lines apart.
llvm::SmallString<256>
getModuleFilesDirectory(PathRef ModuleUnitFileName,
                        const tooling::CompileCommand &Cmd,
                        const GlobalCompilationDatabase &CDB) {
  llvm::SmallString<256> Result = getModuleCacheRoot(ModuleUnitFileName, CDB);
  llvm::sys::path::append(Result,
                          getModuleUnitSourceDirectoryName(ModuleUnitFileName),
                          getCompileCommandStringHash(Cmd));
  return Result;
}

/// Returns the lock file path guarding publication of BMIs for a module unit
/// source. Builders targeting the same source-hash serialize through this path.
llvm::SmallString<256>
getModuleSourceHashLockPath(PathRef ModuleUnitFileName,
                            const GlobalCompilationDatabase &CDB) {
  llvm::SmallString<256> Result =
      getModuleCacheLocksDirectory(ModuleUnitFileName, CDB);
  llvm::sys::path::append(Result,
                          getModuleUnitSourcePathHash(ModuleUnitFileName));
  return Result;
}

/// Returns a unique temporary path used to stage a BMI before atomically
/// publishing it to the stable cache path.
llvm::SmallString<256> getTemporaryModuleFilePath(PathRef ModuleFilePath) {
  llvm::SmallString<256> ResultPattern(ModuleFilePath);
  ResultPattern.append(".tmp-%%-%%-%%-%%-%%-%%");
  llvm::SmallString<256> Result;
  llvm::sys::fs::createUniquePath(ResultPattern, Result,
                                  /*MakeAbsolute=*/false);
  return Result;
}

std::string getModuleFileVersionTimestamp() {
  const auto Now = std::chrono::system_clock::now();
  const auto Micros = std::chrono::duration_cast<std::chrono::microseconds>(
                          Now.time_since_epoch()) %
                      std::chrono::seconds(1);
  const std::time_t CalendarTime = std::chrono::system_clock::to_time_t(Now);
  std::tm LocalTime;
#ifdef _WIN32
  localtime_s(&LocalTime, &CalendarTime);
#else
  localtime_r(&CalendarTime, &LocalTime);
#endif

  return llvm::formatv("{0:04}{1:02}{2:02}-{3:02}{4:02}{5:02}-{6:06}",
                       LocalTime.tm_year + 1900, LocalTime.tm_mon + 1,
                       LocalTime.tm_mday, LocalTime.tm_hour, LocalTime.tm_min,
                       LocalTime.tm_sec, Micros.count())
      .str();
}

llvm::SmallString<256>
getCopyOnReadModuleFilePath(PathRef PublishedModuleFile) {
  llvm::SmallString<256> Result(PublishedModuleFile);
  llvm::sys::path::remove_filename(Result);
  llvm::sys::path::append(
      Result,
      llvm::formatv("{0}-{1}{2}", llvm::sys::path::stem(PublishedModuleFile),
                    getModuleFileVersionTimestamp(),
                    llvm::sys::path::extension(PublishedModuleFile))
          .str());
  return Result;
}

/// Ensures the lock anchor file exists before LockFileManager tries to acquire
/// ownership, creating parent directories as needed.
llvm::Error ensureLockAnchorFileExists(PathRef LockPath) {
  llvm::SmallString<256> LockParent(LockPath);
  llvm::sys::path::remove_filename(LockParent);
  if (std::error_code EC = llvm::sys::fs::create_directories(LockParent))
    return llvm::createStringError(llvm::formatv(
        "Failed to create lock directory {0}: {1}", LockParent, EC.message()));

  int FD = -1;
  if (std::error_code EC = llvm::sys::fs::openFileForWrite(
          LockPath, FD, llvm::sys::fs::CD_OpenAlways))
    return llvm::createStringError(llvm::formatv(
        "Failed to open lock file anchor {0}: {1}", LockPath, EC.message()));
  llvm::sys::Process::SafelyCloseFileDescriptor(FD);
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Persistent Module Cache Locking.
//
// Builders targeting the same module-unit source share a source-hash lock.
// This serializes in-place replacement of stale cache entries and final publish
// of the stable BMI path, while still allowing unrelated module sources to be
// built concurrently.
//
//   builder A                builder B
//      |                        |
//      +---- lock(source) ----->|
//      |                        |
//      | build/publish BMI      | wait
//      |                        |
//      +---- unlock ----------->|
//                               | reuse or rebuild
//===----------------------------------------------------------------------===//

/// Serializes publication and in-place replacement of persistent BMIs for a
/// single module-unit source across multiple builders.
class ScopedModuleSourceLock {
public:
  static llvm::Expected<ScopedModuleSourceLock>
  acquire(PathRef ModuleUnitFileName, const GlobalCompilationDatabase &CDB) {
    constexpr auto LockWaitInterval = std::chrono::seconds(10);
    llvm::SmallString<256> LockPath =
        getModuleSourceHashLockPath(ModuleUnitFileName, CDB);
    if (llvm::Error Err = ensureLockAnchorFileExists(LockPath))
      return std::move(Err);

    auto Waited = std::chrono::seconds::zero();

    while (true) {
      auto Lock = std::make_unique<llvm::LockFileManager>(LockPath);
      auto TryLock = Lock->tryLock();
      if (!TryLock)
        return TryLock.takeError();
      if (*TryLock)
        return ScopedModuleSourceLock(std::move(Lock));

      switch (Lock->waitForUnlockFor(LockWaitInterval)) {
      case llvm::WaitForUnlockResult::Success:
      case llvm::WaitForUnlockResult::OwnerDied:
        continue;
      case llvm::WaitForUnlockResult::Timeout:
        Waited += LockWaitInterval;
        log("Still waiting for module lock {0} after {1}s", LockPath,
            Waited.count());
        continue;
      }
      llvm_unreachable("Unhandled lock wait result");
    }
  }

private:
  explicit ScopedModuleSourceLock(std::unique_ptr<llvm::LockFileManager> Lock)
      : Lock(std::move(Lock)) {}

  std::unique_ptr<llvm::LockFileManager> Lock;
};

// Get the stable published module file path under \param ModuleFilesPrefix.
std::string getModuleFilePath(llvm::StringRef ModuleName,
                              PathRef ModuleFilesPrefix) {
  llvm::SmallString<256> ModuleFilePath(ModuleFilesPrefix);
  auto [PrimaryModuleName, PartitionName] = ModuleName.split(':');
  llvm::sys::path::append(ModuleFilePath, PrimaryModuleName);
  if (!PartitionName.empty()) {
    ModuleFilePath.append("-");
    ModuleFilePath.append(PartitionName);
  }

  ModuleFilePath.append(".pcm");
  return std::string(ModuleFilePath);
}

std::string getPublishedModuleFilePath(llvm::StringRef ModuleName,
                                       PathRef ModuleFilesPrefix) {
  return getModuleFilePath(ModuleName, ModuleFilesPrefix);
}

// FailedPrerequisiteModules - stands for the PrerequisiteModules which has
// errors happened during the building process.
class FailedPrerequisiteModules : public PrerequisiteModules {
public:
  ~FailedPrerequisiteModules() override = default;

  // We shouldn't adjust the compilation commands based on
  // FailedPrerequisiteModules.
  void adjustHeaderSearchOptions(HeaderSearchOptions &Options) const override {}

  // FailedPrerequisiteModules can never be reused.
  bool
  canReuse(const CompilerInvocation &CI,
           llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>) const override {
    return false;
  }
};

/// Represents a reference to a module file (*.pcm).
class ModuleFile {
protected:
  ModuleFile(StringRef ModuleName, PathRef ModuleFilePath)
      : ModuleName(ModuleName.str()), ModuleFilePath(ModuleFilePath.str()) {}

public:
  ModuleFile() = delete;

  ModuleFile(const ModuleFile &) = delete;
  ModuleFile operator=(const ModuleFile &) = delete;

  // The move constructor is needed for llvm::SmallVector.
  ModuleFile(ModuleFile &&Other)
      : ModuleName(std::move(Other.ModuleName)),
        ModuleFilePath(std::move(Other.ModuleFilePath)) {
    Other.ModuleName.clear();
    Other.ModuleFilePath.clear();
  }

  ModuleFile &operator=(ModuleFile &&Other) {
    if (this == &Other)
      return *this;

    this->~ModuleFile();
    new (this) ModuleFile(std::move(Other));
    return *this;
  }
  virtual ~ModuleFile() = default;

  StringRef getModuleName() const { return ModuleName; }

  StringRef getModuleFilePath() const { return ModuleFilePath; }

protected:
  std::string ModuleName;
  std::string ModuleFilePath;
};

/// Represents a prebuilt module file which is not owned by us.
class PrebuiltModuleFile : public ModuleFile {
private:
  // private class to make sure the class can only be constructed by member
  // functions.
  struct CtorTag {};

public:
  PrebuiltModuleFile(StringRef ModuleName, PathRef ModuleFilePath, CtorTag)
      : ModuleFile(ModuleName, ModuleFilePath) {}

  static std::shared_ptr<PrebuiltModuleFile> make(StringRef ModuleName,
                                                  PathRef ModuleFilePath) {
    return std::make_shared<PrebuiltModuleFile>(ModuleName, ModuleFilePath,
                                                CtorTag{});
  }
};

//===----------------------------------------------------------------------===//
// Module File Ownership and Reuse.
//
// PrebuiltModuleFile refers to BMIs supplied directly by the compile command.
// BuiltModuleFile refers to BMIs produced by clangd and published into the
// persistent cache. Object lifetime does not control filesystem lifetime for
// BuiltModuleFile; cache files remain on disk for reuse across builders. The
// versioned copies handed to clang for actual reads are owned by
// CopyOnReadModuleFile and are deleted when the last reader releases them.
//
// Copy-on-read keeps the published BMI path stable for future builders while
// avoiding in-place replacement races for active readers. clangd never hands
// the stable cache entry directly to parsing code. Instead, once a published
// BMI is known to be up to date, clangd copies it to a versioned sibling path
// and gives that copy to readers. Rebuilding only mutates the stable cache
// entry; existing readers keep their own immutable copy until the last
// shared_ptr reference drops and the copy-on-read file is deleted.
//
//   compile command ---------> PrebuiltModuleFile
//
//   clangd build -> publish -> BuiltModuleFile ------------> stable cache path
//                                                           (M.pcm)
//                              |
//                              +-> copy for read -> CopyOnReadModuleFile
//                                                   (M-<timestamp>.pcm)
//                                                   -> handed to clang readers
//                                                   -> removed on last release
//
//   later builder -----------> reuse stable cache path ----> copy for read
//===----------------------------------------------------------------------===//

/// Represents a module file built and published by clangd into its persistent
/// cache.
class BuiltModuleFile final : public ModuleFile {
private:
  // private class to make sure the class can only be constructed by member
  // functions.
  struct CtorTag {};

public:
  BuiltModuleFile(StringRef ModuleName, PathRef ModuleFilePath, CtorTag)
      : ModuleFile(ModuleName, ModuleFilePath) {}

  static std::shared_ptr<BuiltModuleFile> make(StringRef ModuleName,
                                               PathRef ModuleFilePath) {
    return std::make_shared<BuiltModuleFile>(ModuleName, ModuleFilePath,
                                             CtorTag{});
  }
};

/// Represents a versioned copy of a published BMI handed to clangd readers.
/// The copy is removed when the last reader releases it.
class CopyOnReadModuleFile final : public ModuleFile {
private:
  struct CtorTag {};

public:
  CopyOnReadModuleFile(StringRef ModuleName, PathRef ModuleFilePath, CtorTag)
      : ModuleFile(ModuleName, ModuleFilePath) {}

  ~CopyOnReadModuleFile() override {
    if (!ModuleFilePath.empty() && !DebugModulesBuilder)
      if (std::error_code EC = llvm::sys::fs::remove(ModuleFilePath))
        vlog("Failed to remove copy-on-read module file {0}: {1}",
             ModuleFilePath, EC.message());
  }

  static std::shared_ptr<CopyOnReadModuleFile> make(StringRef ModuleName,
                                                    PathRef ModuleFilePath) {
    return std::make_shared<CopyOnReadModuleFile>(ModuleName, ModuleFilePath,
                                                  CtorTag{});
  }
};

// ReusablePrerequisiteModules - stands for PrerequisiteModules for which all
// the required modules are built successfully. All the module files
// are owned by the modules builder.
class ReusablePrerequisiteModules : public PrerequisiteModules {
public:
  ReusablePrerequisiteModules() = default;

  ReusablePrerequisiteModules(const ReusablePrerequisiteModules &Other) =
      default;
  ReusablePrerequisiteModules &
  operator=(const ReusablePrerequisiteModules &) = default;
  ReusablePrerequisiteModules(ReusablePrerequisiteModules &&) = delete;
  ReusablePrerequisiteModules
  operator=(ReusablePrerequisiteModules &&) = delete;

  ~ReusablePrerequisiteModules() override = default;

  void adjustHeaderSearchOptions(HeaderSearchOptions &Options) const override {
    // Appending all built module files.
    for (const auto &RequiredModule : RequiredModules)
      Options.PrebuiltModuleFiles.insert_or_assign(
          RequiredModule->getModuleName().str(),
          RequiredModule->getModuleFilePath().str());
  }

  std::string getAsString() const {
    std::string Result;
    llvm::raw_string_ostream OS(Result);
    for (const auto &MF : RequiredModules) {
      OS << "-fmodule-file=" << MF->getModuleName() << "="
         << MF->getModuleFilePath() << " ";
    }
    return Result;
  }

  bool canReuse(const CompilerInvocation &CI,
                llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>) const override;

  bool isModuleUnitBuilt(llvm::StringRef ModuleName) const {
    return BuiltModuleNames.contains(ModuleName);
  }

  void addModuleFile(std::shared_ptr<const ModuleFile> MF) {
    BuiltModuleNames.insert(MF->getModuleName());
    RequiredModules.emplace_back(std::move(MF));
  }

private:
  llvm::SmallVector<std::shared_ptr<const ModuleFile>, 8> RequiredModules;
  // A helper class to speedup the query if a module is built.
  llvm::StringSet<> BuiltModuleNames;
};

bool IsModuleFileUpToDate(PathRef ModuleFilePath,
                          const PrerequisiteModules &RequisiteModules,
                          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) {
  HeaderSearchOptions HSOpts;
  RequisiteModules.adjustHeaderSearchOptions(HSOpts);
  HSOpts.ForceCheckCXX20ModulesInputFiles = true;
  HSOpts.ValidateASTInputFilesContent = true;

  clang::clangd::IgnoreDiagnostics IgnoreDiags;
  DiagnosticOptions DiagOpts;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(*VFS, DiagOpts, &IgnoreDiags,
                                          /*ShouldOwnClient=*/false);

  LangOptions LangOpts;
  LangOpts.SkipODRCheckInGMF = true;

  FileManager FileMgr(FileSystemOptions(), VFS);

  SourceManager SourceMgr(*Diags, FileMgr);

  HeaderSearch HeaderInfo(HSOpts, SourceMgr, *Diags, LangOpts,
                          /*Target=*/nullptr);

  PreprocessorOptions PPOpts;
  TrivialModuleLoader ModuleLoader;
  Preprocessor PP(PPOpts, *Diags, LangOpts, SourceMgr, HeaderInfo,
                  ModuleLoader);

  std::shared_ptr<ModuleCache> ModCache = createCrossProcessModuleCache();
  PCHContainerOperations PCHOperations;
  CodeGenOptions CodeGenOpts;
  ASTReader Reader(
      PP, *ModCache, /*ASTContext=*/nullptr, PCHOperations.getRawReader(),
      CodeGenOpts, {},
      /*isysroot=*/"",
      /*DisableValidationKind=*/DisableValidationForModuleKind::None,
      /*AllowASTWithCompilerErrors=*/false,
      /*AllowConfigurationMismatch=*/false,
      /*ValidateSystemInputs=*/false,
      /*ForceValidateUserInputs=*/true,
      /*ValidateASTInputFilesContent=*/true);

  // We don't need any listener here. By default it will use a validator
  // listener.
  Reader.setListener(nullptr);

  // Use ARR_OutOfDate so that ReadAST returns OutOfDate instead of Failure
  // when input files are modified. This allows us to detect staleness
  // without treating it as a hard error.
  // ReadAST will validate all input files internally and return OutOfDate
  // if any file is modified.
  return Reader.ReadAST(ModuleFileName::makeExplicit(ModuleFilePath),
                        serialization::MK_MainFile, SourceLocation(),
                        ASTReader::ARR_OutOfDate) == ASTReader::Success;
}

bool IsModuleFilesUpToDate(
    llvm::SmallVector<PathRef> ModuleFilePaths,
    const PrerequisiteModules &RequisiteModules,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) {
  return llvm::all_of(
      ModuleFilePaths, [&RequisiteModules, VFS](auto ModuleFilePath) {
        return IsModuleFileUpToDate(ModuleFilePath, RequisiteModules, VFS);
      });
}

/// Builds a BMI into a temporary file and publishes it to `ModuleFilePath`.
/// If another builder wins the publish race first, reports that through
/// `PublishedExistingModuleFile` so the caller can validate and reuse it.
llvm::Expected<std::shared_ptr<BuiltModuleFile>>
buildModuleFile(llvm::StringRef ModuleName, PathRef ModuleUnitFileName,
                tooling::CompileCommand Cmd, PathRef ModuleFilePath,
                const ThreadsafeFS &TFS,
                const ReusablePrerequisiteModules &BuiltModuleFiles,
                bool &PublishedExistingModuleFile) {
  PublishedExistingModuleFile = false;
  llvm::SmallString<256> ModuleFilesPrefix(ModuleFilePath);
  llvm::sys::path::remove_filename(ModuleFilesPrefix);
  if (std::error_code EC = llvm::sys::fs::create_directories(ModuleFilesPrefix))
    return llvm::createStringError(
        llvm::formatv("Failed to create module cache directory {0}: {1}",
                      ModuleFilesPrefix, EC.message()));

  llvm::SmallString<256> TemporaryModuleFilePath =
      getTemporaryModuleFilePath(ModuleFilePath);
  auto RemoveTemporaryModuleFile = llvm::scope_exit([&] {
    if (!TemporaryModuleFilePath.empty() && !DebugModulesBuilder)
      llvm::sys::fs::remove(TemporaryModuleFilePath);
  });
  (void)RemoveTemporaryModuleFile;

  Cmd.Output = TemporaryModuleFilePath.str().str();

  ParseInputs Inputs;
  Inputs.TFS = &TFS;
  Inputs.CompileCommand = std::move(Cmd);

  IgnoreDiagnostics IgnoreDiags;
  auto CI = buildCompilerInvocation(Inputs, IgnoreDiags);
  if (!CI)
    return llvm::createStringError("Failed to build compiler invocation");

  auto FS = Inputs.TFS->view(Inputs.CompileCommand.Directory);
  auto Buf = FS->getBufferForFile(Inputs.CompileCommand.Filename);
  if (!Buf)
    return llvm::createStringError("Failed to create buffer");

  // In clang's driver, we will suppress the check for ODR violation in GMF.
  // See the implementation of RenderModulesOptions in Clang.cpp.
  CI->getLangOpts().SkipODRCheckInGMF = true;

  // Hash the contents of input files and store the hash value to the BMI files.
  // So that we can check if the files are still valid when we want to reuse the
  // BMI files.
  CI->getHeaderSearchOpts().ValidateASTInputFilesContent = true;

  BuiltModuleFiles.adjustHeaderSearchOptions(CI->getHeaderSearchOpts());

  CI->getFrontendOpts().OutputFile = Inputs.CompileCommand.Output;
  auto Clang =
      prepareCompilerInstance(std::move(CI), /*Preamble=*/nullptr,
                              std::move(*Buf), std::move(FS), IgnoreDiags);
  if (!Clang)
    return llvm::createStringError("Failed to prepare compiler instance");

  GenerateReducedModuleInterfaceAction Action;
  Clang->ExecuteAction(Action);

  if (Clang->getDiagnostics().hasErrorOccurred()) {
    std::string Cmds;
    for (const auto &Arg : Inputs.CompileCommand.CommandLine) {
      if (!Cmds.empty())
        Cmds += " ";
      Cmds += Arg;
    }

    clangd::vlog("Failed to compile {0} with command: {1}", ModuleUnitFileName,
                 Cmds);

    std::string BuiltModuleFilesStr = BuiltModuleFiles.getAsString();
    if (!BuiltModuleFilesStr.empty())
      clangd::vlog("The actual used module files built by clangd is {0}",
                   BuiltModuleFilesStr);

    return llvm::createStringError(
        llvm::formatv("Failed to compile {0}. Use '--log=verbose' to view "
                      "detailed failure reasons. It is helpful to use "
                      "'--debug-modules-builder' flag to keep the clangd's "
                      "built module files to reproduce the failure for "
                      "debugging. Remember to remove them after debugging.",
                      ModuleUnitFileName));
  }

  if (std::error_code EC =
          llvm::sys::fs::rename(TemporaryModuleFilePath, ModuleFilePath)) {
    if (!llvm::sys::fs::exists(ModuleFilePath))
      return llvm::createStringError(
          llvm::formatv("Failed to publish module file {0}: {1}",
                        ModuleFilePath, EC.message()));
    // Another builder already published the stable cache entry. Drop our
    // staged BMI and let the caller revalidate the published path.
    PublishedExistingModuleFile = true;
  } else {
    // Rename consumed the staging file into the stable cache path. Clear it so
    // the scope-exit cleanup does not try to remove the published BMI.
    TemporaryModuleFilePath.clear();
  }

  return BuiltModuleFile::make(ModuleName, ModuleFilePath);
}

llvm::Expected<std::shared_ptr<CopyOnReadModuleFile>>
copyModuleFileForRead(llvm::StringRef ModuleName,
                      PathRef PublishedModuleFilePath) {
  llvm::SmallString<256> VersionedModuleFilePath =
      getCopyOnReadModuleFilePath(PublishedModuleFilePath);
  if (std::error_code EC = llvm::sys::fs::copy_file(PublishedModuleFilePath,
                                                    VersionedModuleFilePath))
    return llvm::createStringError(llvm::formatv(
        "Failed to copy module file {0} to {1}: {2}", PublishedModuleFilePath,
        VersionedModuleFilePath, EC.message()));
  return CopyOnReadModuleFile::make(ModuleName, VersionedModuleFilePath);
}

bool ReusablePrerequisiteModules::canReuse(
    const CompilerInvocation &CI,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) const {
  if (RequiredModules.empty())
    return true;

  llvm::SmallVector<llvm::StringRef> BMIPaths;
  for (auto &MF : RequiredModules)
    BMIPaths.push_back(MF->getModuleFilePath());
  return IsModuleFilesUpToDate(BMIPaths, *this, VFS);
}

//===----------------------------------------------------------------------===//
// In-Memory Module File Cache.
//
// This cache deduplicates BMIs within a single builder instance. Its key
// mirrors the persistent cache layout: module name, module-unit source, and
// compile command hash. That prevents a builder from reusing a BMI built under
// a different command line.
//
//   (module name,
//    module-unit source,
//    command hash)
//          |
//          v
//      ModuleFileCache
//          |
//          +-- hit  -> reuse in current builder
//          |
//          +-- miss -> probe persistent cache / rebuild
//===----------------------------------------------------------------------===//

/// In-memory cache for module files built by clangd. Entries are keyed by
/// module name, module-unit source, and compile-command hash so persistent BMI
/// variants do not collide.
class ModuleFileCache {
public:
  ModuleFileCache(const GlobalCompilationDatabase &CDB) : CDB(CDB) {}
  const GlobalCompilationDatabase &getCDB() const { return CDB; }

  std::shared_ptr<const ModuleFile> getModule(StringRef ModuleName,
                                              PathRef ModuleUnitSource,
                                              llvm::StringRef CommandHash);

  void add(StringRef ModuleName, PathRef ModuleUnitSource,
           llvm::StringRef CommandHash,
           std::shared_ptr<const ModuleFile> ModuleFile) {
    std::lock_guard<std::mutex> Lock(ModuleFilesMutex);
    ModuleFiles[cacheKey(ModuleName, ModuleUnitSource, CommandHash)] =
        ModuleFile;
  }

  void remove(StringRef ModuleName, PathRef ModuleUnitSource,
              llvm::StringRef CommandHash);

private:
  static std::string cacheKey(StringRef ModuleName, PathRef ModuleUnitSource,
                              llvm::StringRef CommandHash) {
    std::string Key;
    Key.reserve(ModuleName.size() + ModuleUnitSource.size() +
                CommandHash.size() + 2);
    Key.append(ModuleName);
    Key.push_back('\0');
    Key.append(maybeCaseFoldPath(ModuleUnitSource));
    Key.push_back('\0');
    Key.append(CommandHash);
    return Key;
  }

  const GlobalCompilationDatabase &CDB;

  llvm::StringMap<std::weak_ptr<const ModuleFile>> ModuleFiles;
  std::mutex ModuleFilesMutex;
};

std::shared_ptr<const ModuleFile>
ModuleFileCache::getModule(StringRef ModuleName, PathRef ModuleUnitSource,
                           llvm::StringRef CommandHash) {
  std::lock_guard<std::mutex> Lock(ModuleFilesMutex);

  auto Iter =
      ModuleFiles.find(cacheKey(ModuleName, ModuleUnitSource, CommandHash));
  if (Iter == ModuleFiles.end())
    return nullptr;

  if (auto Res = Iter->second.lock())
    return Res;

  ModuleFiles.erase(Iter);
  return nullptr;
}

void ModuleFileCache::remove(StringRef ModuleName, PathRef ModuleUnitSource,
                             llvm::StringRef CommandHash) {
  std::lock_guard<std::mutex> Lock(ModuleFilesMutex);
  ModuleFiles.erase(cacheKey(ModuleName, ModuleUnitSource, CommandHash));
}

class ModuleNameToSourceCache {
public:
  std::string getUniqueSourceForModuleName(llvm::StringRef ModuleName) {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    auto Iter = ModuleNameToUniqueSourceCache.find(ModuleName);
    if (Iter != ModuleNameToUniqueSourceCache.end())
      return Iter->second;
    return "";
  }

  void addUniqueEntry(llvm::StringRef ModuleName, PathRef Source) {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    ModuleNameToUniqueSourceCache[ModuleName] = Source.str();
  }

  void eraseUniqueEntry(llvm::StringRef ModuleName) {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    ModuleNameToUniqueSourceCache.erase(ModuleName);
  }

  std::string getMultipleSourceForModuleName(llvm::StringRef ModuleName,
                                             PathRef RequiredSrcFile) {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    auto Outer = ModuleNameToMultipleSourceCache.find(ModuleName);
    if (Outer == ModuleNameToMultipleSourceCache.end())
      return "";
    auto Inner = Outer->second.find(maybeCaseFoldPath(RequiredSrcFile));
    if (Inner == Outer->second.end())
      return "";
    return Inner->second;
  }

  void addMultipleEntry(llvm::StringRef ModuleName, PathRef RequiredSrcFile,
                        PathRef Source) {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    ModuleNameToMultipleSourceCache[ModuleName]
                                   [maybeCaseFoldPath(RequiredSrcFile)] =
                                       Source.str();
  }

  void eraseMultipleEntry(llvm::StringRef ModuleName, PathRef RequiredSrcFile) {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    auto Outer = ModuleNameToMultipleSourceCache.find(ModuleName);
    if (Outer == ModuleNameToMultipleSourceCache.end())
      return;
    Outer->second.erase(maybeCaseFoldPath(RequiredSrcFile));
    if (Outer->second.empty())
      ModuleNameToMultipleSourceCache.erase(Outer);
  }

private:
  std::mutex CacheMutex;
  llvm::StringMap<std::string> ModuleNameToUniqueSourceCache;

  // Map from module name to a map from required source to module unit source
  // which declares the corresponding module name.
  // This looks inefficiency. We can only assume there won't too many duplicated
  // module names with different module units in a project.
  llvm::StringMap<llvm::StringMap<std::string>> ModuleNameToMultipleSourceCache;
};

class CachingProjectModules : public ProjectModules {
public:
  CachingProjectModules(std::unique_ptr<ProjectModules> MDB,
                        ModuleNameToSourceCache &Cache)
      : MDB(std::move(MDB)), Cache(Cache) {
    assert(this->MDB && "CachingProjectModules should only be created with a "
                        "valid underlying ProjectModules");
  }

  std::vector<std::string> getRequiredModules(PathRef File) override {
    return MDB->getRequiredModules(File);
  }

  std::string getModuleNameForSource(PathRef File) override {
    return MDB->getModuleNameForSource(File);
  }

  ModuleNameState getModuleNameState(llvm::StringRef ModuleName) override {
    return MDB->getModuleNameState(ModuleName);
  }

  std::string getSourceForModuleName(llvm::StringRef ModuleName,
                                     PathRef RequiredSrcFile) override {
    auto ModuleState = MDB->getModuleNameState(ModuleName);

    if (ModuleState == ModuleNameState::Multiple) {
      std::string CachedResult =
          Cache.getMultipleSourceForModuleName(ModuleName, RequiredSrcFile);

      // Verify Cached Result by seeing if the source declaring the same module
      // as we query.
      if (!CachedResult.empty()) {
        std::string ModuleNameOfCachedSource =
            MDB->getModuleNameForSource(CachedResult);
        if (ModuleNameOfCachedSource == ModuleName)
          return CachedResult;

        // Cached Result is invalid. Clear it.
        Cache.eraseMultipleEntry(ModuleName, RequiredSrcFile);
      }

      auto Result = MDB->getSourceForModuleName(ModuleName, RequiredSrcFile);
      if (!Result.empty())
        Cache.addMultipleEntry(ModuleName, RequiredSrcFile, Result);
      return Result;
    }

    // For unknown module name state, assume it is unique. This may give user
    // higher usability.
    assert(ModuleState == ModuleNameState::Unique ||
           ModuleState == ModuleNameState::Unknown);
    std::string CachedResult = Cache.getUniqueSourceForModuleName(ModuleName);

    // Verify Cached Result by seeing if the source declaring the same module
    // as we query.
    if (!CachedResult.empty()) {
      std::string ModuleNameOfCachedSource =
          MDB->getModuleNameForSource(CachedResult);
      if (ModuleNameOfCachedSource == ModuleName)
        return CachedResult;

      // Cached Result is invalid. Clear it.
      Cache.eraseUniqueEntry(ModuleName);
    }

    auto Result = MDB->getSourceForModuleName(ModuleName, RequiredSrcFile);
    if (!Result.empty())
      Cache.addUniqueEntry(ModuleName, Result);

    return Result;
  }

private:
  std::unique_ptr<ProjectModules> MDB;
  ModuleNameToSourceCache &Cache;
};

/// Collect the directly and indirectly required module names for \param
/// ModuleName in topological order. The \param ModuleName is guaranteed to
/// be the last element in \param ModuleNames.
llvm::SmallVector<std::string> getAllRequiredModules(PathRef RequiredSource,
                                                     CachingProjectModules &MDB,
                                                     StringRef ModuleName) {
  llvm::SmallVector<std::string> ModuleNames;
  llvm::StringSet<> ModuleNamesSet;

  auto VisitDeps = [&](StringRef ModuleName, auto Visitor) -> void {
    ModuleNamesSet.insert(ModuleName);

    for (StringRef RequiredModuleName : MDB.getRequiredModules(
             MDB.getSourceForModuleName(ModuleName, RequiredSource)))
      if (ModuleNamesSet.insert(RequiredModuleName).second)
        Visitor(RequiredModuleName, Visitor);

    ModuleNames.push_back(ModuleName.str());
  };
  VisitDeps(ModuleName, VisitDeps);

  return ModuleNames;
}

/// Collects cache roots to scan during constructor-time GC.
/// Scans one cache root and returns all `.pcm` files under it.
std::vector<std::string> collectModuleFiles(PathRef CacheRoot) {
  std::vector<std::string> Result;
  std::error_code EC;
  for (llvm::sys::fs::recursive_directory_iterator It(CacheRoot, EC), End;
       It != End && !EC; It.increment(EC)) {
    if (llvm::sys::path::extension(It->path()) != ".pcm")
      continue;
    Result.push_back(It->path());
  }
  if (EC)
    log("Failed to scan module cache directory {0}: {1}", CacheRoot,
        EC.message());
  return Result;
}

/// Performs one GC pass over a persistent module cache root.
void garbageCollectModuleCache(PathRef CacheRoot) {
  for (const auto &ModuleFilePath : collectModuleFiles(CacheRoot)) {
    llvm::sys::fs::file_status Status;
    if (std::error_code EC = llvm::sys::fs::status(ModuleFilePath, Status)) {
      log("Failed to stat cached module file {0} for GC: {1}", ModuleFilePath,
          EC.message());
      continue;
    }

    llvm::sys::TimePoint<> LastAccess = Status.getLastAccessedTime();
    llvm::sys::TimePoint<> Now = std::chrono::system_clock::now();
    if (LastAccess > Now)
      continue;
    auto Age =
        std::chrono::duration_cast<std::chrono::seconds>(Now - LastAccess);
    auto Threshold =
        std::chrono::seconds(VersionedModuleFileGCThresholdSeconds);
    if (Age <= Threshold)
      continue;

    if (!llvm::sys::fs::exists(ModuleFilePath))
      continue;

    constexpr llvm::StringLiteral Reason = "file older than GC threshold";
    if (std::error_code EC = llvm::sys::fs::remove(ModuleFilePath)) {
      log("Failed to remove cached module file {0} ({1}): {2}", ModuleFilePath,
          Reason, EC.message());
      continue;
    }
    log("Removed cached module file {0} ({1})", ModuleFilePath, Reason);
  }
}

} // namespace

class ModulesBuilder::ModulesBuilderImpl {
public:
  ModulesBuilderImpl(const GlobalCompilationDatabase &CDB) : Cache(CDB) {}

  ModuleNameToSourceCache &getProjectModulesCache() {
    return ProjectModulesCache;
  }
  const GlobalCompilationDatabase &getCDB() const { return Cache.getCDB(); }

  llvm::Error
  getOrBuildModuleFile(PathRef RequiredSource, StringRef ModuleName,
                       const ThreadsafeFS &TFS, CachingProjectModules &MDB,
                       ReusablePrerequisiteModules &BuiltModuleFiles);

private:
  /// Try to get prebuilt module files from the compilation database.
  void getPrebuiltModuleFile(StringRef ModuleName, PathRef ModuleUnitFileName,
                             const ThreadsafeFS &TFS,
                             ReusablePrerequisiteModules &BuiltModuleFiles);

  /// Runs GC once for the cache root owning a project root.
  void garbageCollectModuleCacheForProjectRoot(PathRef ProjectRoot);

  ModuleFileCache Cache;
  ModuleNameToSourceCache ProjectModulesCache;
  std::mutex GarbageCollectedProjectRootsMutex;
  llvm::StringSet<> GarbageCollectedProjectRoots;
};

void ModulesBuilder::ModulesBuilderImpl::
    garbageCollectModuleCacheForProjectRoot(PathRef ProjectRoot) {
  if (ProjectRoot.empty())
    return;
  std::string NormalizedProjectRoot = normalizePathForCache(ProjectRoot);
  {
    // If the project root lives in GarbageCollectedProjectRoots, it implies
    // we've already started GC on the cache root.
    std::lock_guard<std::mutex> Lock(GarbageCollectedProjectRootsMutex);
    if (!GarbageCollectedProjectRoots.insert(NormalizedProjectRoot).second)
      return;
  }

  llvm::SmallString<256> CacheRoot(ProjectRoot);
  llvm::sys::path::append(CacheRoot, ".cache", "clangd", "modules");
  log("Running GC pass for clangd built module files under {0} with age "
      "threshold {1} seconds (adjust with --modules-builder-versioned-gc-"
      "threshold-seconds)",
      CacheRoot, VersionedModuleFileGCThresholdSeconds);
  garbageCollectModuleCache(CacheRoot);
  log("Done running GC pass for clangd built module files under {0}",
      CacheRoot);
}

void ModulesBuilder::ModulesBuilderImpl::getPrebuiltModuleFile(
    StringRef ModuleName, PathRef ModuleUnitFileName, const ThreadsafeFS &TFS,
    ReusablePrerequisiteModules &BuiltModuleFiles) {
  auto Cmd = getCDB().getCompileCommand(ModuleUnitFileName);
  if (!Cmd)
    return;

  ParseInputs Inputs;
  Inputs.TFS = &TFS;
  Inputs.CompileCommand = std::move(*Cmd);

  IgnoreDiagnostics IgnoreDiags;
  auto CI = buildCompilerInvocation(Inputs, IgnoreDiags);
  if (!CI)
    return;

  // We don't need to check if the module files are in ModuleCache or adding
  // them to the module cache. As even if the module files are in the module
  // cache, we still need to validate them. And it looks not helpful to add them
  // to the module cache, since we may always try to get the prebuilt module
  // files before building the module files by ourselves.
  for (auto &[ModuleName, ModuleFilePath] :
       CI->getHeaderSearchOpts().PrebuiltModuleFiles) {
    if (BuiltModuleFiles.isModuleUnitBuilt(ModuleName))
      continue;

    // Convert relative path to absolute path based on the compilation directory
    llvm::SmallString<256> AbsoluteModuleFilePath;
    if (llvm::sys::path::is_relative(ModuleFilePath)) {
      AbsoluteModuleFilePath = Inputs.CompileCommand.Directory;
      llvm::sys::path::append(AbsoluteModuleFilePath, ModuleFilePath);
    } else
      AbsoluteModuleFilePath = ModuleFilePath;

    if (IsModuleFileUpToDate(AbsoluteModuleFilePath, BuiltModuleFiles,
                             TFS.view(std::nullopt))) {
      log("Reusing prebuilt module file {0} of module {1} for {2}",
          AbsoluteModuleFilePath, ModuleName, ModuleUnitFileName);
      BuiltModuleFiles.addModuleFile(
          PrebuiltModuleFile::make(ModuleName, AbsoluteModuleFilePath));
    }
  }
}

llvm::Error ModulesBuilder::ModulesBuilderImpl::getOrBuildModuleFile(
    PathRef RequiredSource, StringRef ModuleName, const ThreadsafeFS &TFS,
    CachingProjectModules &MDB, ReusablePrerequisiteModules &BuiltModuleFiles) {
  if (BuiltModuleFiles.isModuleUnitBuilt(ModuleName))
    return llvm::Error::success();

  std::string ModuleUnitFileName =
      MDB.getSourceForModuleName(ModuleName, RequiredSource);
  /// It is possible that we're meeting third party modules (modules whose
  /// source are not in the project. e.g, the std module may be a third-party
  /// module for most project) or something wrong with the implementation of
  /// ProjectModules.
  /// FIXME: How should we treat third party modules here? If we want to ignore
  /// third party modules, we should return true instead of false here.
  /// Currently we simply bail out.
  if (ModuleUnitFileName.empty())
    return llvm::createStringError(
        llvm::formatv("Don't get the module unit for module {0}", ModuleName));

  /// Try to get prebuilt module files from the compilation database first. This
  /// helps to avoid building the module files that are already built by the
  /// compiler.
  getPrebuiltModuleFile(ModuleName, ModuleUnitFileName, TFS, BuiltModuleFiles);

  // Get Required modules in topological order.
  auto ReqModuleNames = getAllRequiredModules(RequiredSource, MDB, ModuleName);
  for (llvm::StringRef ReqModuleName : ReqModuleNames) {
    if (BuiltModuleFiles.isModuleUnitBuilt(ReqModuleName))
      continue;

    std::string ReqFileName =
        MDB.getSourceForModuleName(ReqModuleName, RequiredSource);
    auto Cmd = getCDB().getCompileCommand(ReqFileName);
    if (!Cmd)
      return llvm::createStringError(
          llvm::formatv("No compile command for {0}", ReqFileName));
    if (auto PI = getCDB().getProjectInfo(ReqFileName);
        PI && !PI->SourceRoot.empty())
      garbageCollectModuleCacheForProjectRoot(PI->SourceRoot);

    const std::string CommandHash = getCompileCommandStringHash(*Cmd);
    const std::string PublishedModuleFilePath = getPublishedModuleFilePath(
        ReqModuleName, getModuleFilesDirectory(ReqFileName, *Cmd, getCDB()));

    // Keep the source-scoped lock while probing and validating cached BMIs so
    // stale-file replacement and final publication stay serialized.
    auto SourceLock = ScopedModuleSourceLock::acquire(ReqFileName, getCDB());
    if (!SourceLock)
      return SourceLock.takeError();

    std::shared_ptr<const ModuleFile> Cached =
        Cache.getModule(ReqModuleName, ReqFileName, CommandHash);

    if (Cached) {
      if (IsModuleFileUpToDate(Cached->getModuleFilePath(), BuiltModuleFiles,
                               TFS.view(std::nullopt))) {
        log("Reusing module {0} from {1}", ReqModuleName,
            Cached->getModuleFilePath());
        BuiltModuleFiles.addModuleFile(std::move(Cached));
        continue;
      }
      Cache.remove(ReqModuleName, ReqFileName, CommandHash);
    }

    if (llvm::sys::fs::exists(PublishedModuleFilePath)) {
      if (IsModuleFileUpToDate(PublishedModuleFilePath, BuiltModuleFiles,
                               TFS.view(std::nullopt))) {
        log("Reusing persistent module {0} from {1}", ReqModuleName,
            PublishedModuleFilePath);
        auto Materialized =
            copyModuleFileForRead(ReqModuleName, PublishedModuleFilePath);
        if (llvm::Error Err = Materialized.takeError())
          return Err;
        Cache.add(ReqModuleName, ReqFileName, CommandHash, *Materialized);
        BuiltModuleFiles.addModuleFile(std::move(*Materialized));
        continue;
      }

      // The persistent module file is stale. Remove it and build a new one.
      std::error_code EC = llvm::sys::fs::remove(PublishedModuleFilePath);
      if (EC)
        return llvm::createStringError(
            llvm::formatv("Failed to remove stale module file {0}: {1}",
                          PublishedModuleFilePath, EC.message()));
    }

    bool PublishedExistingModuleFile = false;
    llvm::Expected<std::shared_ptr<BuiltModuleFile>> MF = buildModuleFile(
        ReqModuleName, ReqFileName, std::move(*Cmd), PublishedModuleFilePath,
        TFS, BuiltModuleFiles, PublishedExistingModuleFile);
    if (llvm::Error Err = MF.takeError())
      return Err;

    if (PublishedExistingModuleFile &&
        !IsModuleFileUpToDate(PublishedModuleFilePath, BuiltModuleFiles,
                              TFS.view(std::nullopt))) {
      return llvm::createStringError(
          llvm::formatv("Published module file {0} is stale after lock wait",
                        PublishedModuleFilePath));
    }

    auto Materialized =
        copyModuleFileForRead(ReqModuleName, PublishedModuleFilePath);
    if (llvm::Error Err = Materialized.takeError())
      return Err;

    log("Built module {0} to {1}", ReqModuleName,
        (*Materialized)->getModuleFilePath());
    Cache.add(ReqModuleName, ReqFileName, CommandHash, *Materialized);
    BuiltModuleFiles.addModuleFile(std::move(*Materialized));
  }

  return llvm::Error::success();
}

bool ModulesBuilder::hasRequiredModules(PathRef File) {
  std::unique_ptr<ProjectModules> MDB = Impl->getCDB().getProjectModules(File);
  if (!MDB)
    return false;

  CachingProjectModules CachedMDB(std::move(MDB),
                                  Impl->getProjectModulesCache());
  return !CachedMDB.getRequiredModules(File).empty();
}

std::unique_ptr<PrerequisiteModules>
ModulesBuilder::buildPrerequisiteModulesFor(PathRef File,
                                            const ThreadsafeFS &TFS) {
  std::unique_ptr<ProjectModules> MDB = Impl->getCDB().getProjectModules(File);
  if (!MDB) {
    elog("Failed to get Project Modules information for {0}", File);
    return std::make_unique<FailedPrerequisiteModules>();
  }
  CachingProjectModules CachedMDB(std::move(MDB),
                                  Impl->getProjectModulesCache());

  std::vector<std::string> RequiredModuleNames =
      CachedMDB.getRequiredModules(File);
  if (RequiredModuleNames.empty())
    return std::make_unique<ReusablePrerequisiteModules>();

  auto RequiredModules = std::make_unique<ReusablePrerequisiteModules>();
  for (llvm::StringRef RequiredModuleName : RequiredModuleNames) {
    // Return early if there is any error.
    if (llvm::Error Err = Impl->getOrBuildModuleFile(
            File, RequiredModuleName, TFS, CachedMDB, *RequiredModules.get())) {
      elog("Failed to build module {0}; due to {1}", RequiredModuleName,
           toString(std::move(Err)));
      return std::make_unique<FailedPrerequisiteModules>();
    }
  }

  return std::move(RequiredModules);
}

ModulesBuilder::ModulesBuilder(const GlobalCompilationDatabase &CDB) {
  Impl = std::make_unique<ModulesBuilderImpl>(CDB);
}

ModulesBuilder::~ModulesBuilder() {}

} // namespace clangd
} // namespace clang
