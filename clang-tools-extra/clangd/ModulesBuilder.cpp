//===----------------- ModulesBuilder.cpp ------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ModulesBuilder.h"
#include "Compiler.h"
#include "support/Logger.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ModuleCache.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"

#include <queue>

namespace clang {
namespace clangd {

namespace {

llvm::cl::opt<bool> DebugModulesBuilder(
    "debug-modules-builder",
    llvm::cl::desc("Don't remove clangd's built module files for debugging. "
                   "Remember to remove them later after debugging."),
    llvm::cl::init(false));

// Create a path to store module files. Generally it should be:
//
//   {TEMP_DIRS}/clangd/module_files/{hashed-file-name}-%%-%%-%%-%%-%%-%%/.
//
// {TEMP_DIRS} is the temporary directory for the system, e.g., "/var/tmp"
// or "C:/TEMP".
//
// '%%' means random value to make the generated path unique.
//
// \param MainFile is used to get the root of the project from global
// compilation database.
//
// TODO: Move these module fils out of the temporary directory if the module
// files are persistent.
llvm::SmallString<256> getUniqueModuleFilesPath(PathRef MainFile) {
  llvm::SmallString<128> HashedPrefix = llvm::sys::path::filename(MainFile);
  // There might be multiple files with the same name in a project. So appending
  // the hash value of the full path to make sure they won't conflict.
  HashedPrefix += std::to_string(llvm::hash_value(MainFile));

  llvm::SmallString<256> ResultPattern;

  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true,
                                         ResultPattern);

  llvm::sys::path::append(ResultPattern, "clangd");
  llvm::sys::path::append(ResultPattern, "module_files");

  llvm::sys::path::append(ResultPattern, HashedPrefix);

  ResultPattern.append("-%%-%%-%%-%%-%%-%%");

  llvm::SmallString<256> Result;
  llvm::sys::fs::createUniquePath(ResultPattern, Result,
                                  /*MakeAbsolute=*/false);

  llvm::sys::fs::create_directories(Result);
  return Result;
}

// Get a unique module file path under \param ModuleFilesPrefix.
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

// FailedPrerequisiteModules - stands for the PrerequisiteModules which has
// errors happened during the building process.
class FailedPrerequisiteModules : public PrerequisiteModules {
public:
  ~FailedPrerequisiteModules() override = default;

  // We shouldn't adjust the compilation commands based on
  // FailedPrerequisiteModules.
  void adjustHeaderSearchOptions(HeaderSearchOptions &Options) const override {
  }

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

/// Represents a module file built by us. We're responsible to remove it.
class BuiltModuleFile : public ModuleFile {
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

  virtual ~BuiltModuleFile() {
    if (!ModuleFilePath.empty() && !DebugModulesBuilder)
      llvm::sys::fs::remove(ModuleFilePath);
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

  IntrusiveRefCntPtr<ModuleCache> ModCache = createCrossProcessModuleCache();
  PCHContainerOperations PCHOperations;
  CodeGenOptions CodeGenOpts;
  ASTReader Reader(PP, *ModCache, /*ASTContext=*/nullptr,
                   PCHOperations.getRawReader(), CodeGenOpts, {});

  // We don't need any listener here. By default it will use a validator
  // listener.
  Reader.setListener(nullptr);

  if (Reader.ReadAST(ModuleFilePath, serialization::MK_MainFile,
                     SourceLocation(),
                     ASTReader::ARR_None) != ASTReader::Success)
    return false;

  bool UpToDate = true;
  Reader.getModuleManager().visit([&](serialization::ModuleFile &MF) -> bool {
    Reader.visitInputFiles(
        MF, /*IncludeSystem=*/false, /*Complain=*/false,
        [&](const serialization::InputFile &IF, bool isSystem) {
          if (!IF.getFile() || IF.isOutOfDate())
            UpToDate = false;
        });
    return !UpToDate;
  });
  return UpToDate;
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

/// Build a module file for module with `ModuleName`. The information of built
/// module file are stored in \param BuiltModuleFiles.
llvm::Expected<std::shared_ptr<BuiltModuleFile>>
buildModuleFile(llvm::StringRef ModuleName, PathRef ModuleUnitFileName,
                const GlobalCompilationDatabase &CDB, const ThreadsafeFS &TFS,
                const ReusablePrerequisiteModules &BuiltModuleFiles) {
  // Try cheap operation earlier to boil-out cheaply if there are problems.
  auto Cmd = CDB.getCompileCommand(ModuleUnitFileName);
  if (!Cmd)
    return llvm::createStringError(
        llvm::formatv("No compile command for {0}", ModuleUnitFileName));

  llvm::SmallString<256> ModuleFilesPrefix =
      getUniqueModuleFilesPath(ModuleUnitFileName);

  Cmd->Output = getModuleFilePath(ModuleName, ModuleFilesPrefix);

  ParseInputs Inputs;
  Inputs.TFS = &TFS;
  Inputs.CompileCommand = std::move(*Cmd);

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

  return BuiltModuleFile::make(ModuleName, Inputs.CompileCommand.Output);
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

class ModuleFileCache {
public:
  ModuleFileCache(const GlobalCompilationDatabase &CDB) : CDB(CDB) {}
  const GlobalCompilationDatabase &getCDB() const { return CDB; }

  std::shared_ptr<const ModuleFile> getModule(StringRef ModuleName);

  void add(StringRef ModuleName, std::shared_ptr<const ModuleFile> ModuleFile) {
    std::lock_guard<std::mutex> Lock(ModuleFilesMutex);

    ModuleFiles[ModuleName] = ModuleFile;
  }

  void remove(StringRef ModuleName);

private:
  const GlobalCompilationDatabase &CDB;

  llvm::StringMap<std::weak_ptr<const ModuleFile>> ModuleFiles;
  // Mutex to guard accesses to ModuleFiles.
  std::mutex ModuleFilesMutex;
};

std::shared_ptr<const ModuleFile>
ModuleFileCache::getModule(StringRef ModuleName) {
  std::lock_guard<std::mutex> Lock(ModuleFilesMutex);

  auto Iter = ModuleFiles.find(ModuleName);
  if (Iter == ModuleFiles.end())
    return nullptr;

  if (auto Res = Iter->second.lock())
    return Res;

  ModuleFiles.erase(Iter);
  return nullptr;
}

void ModuleFileCache::remove(StringRef ModuleName) {
  std::lock_guard<std::mutex> Lock(ModuleFilesMutex);

  ModuleFiles.erase(ModuleName);
}

class ModuleNameToSourceCache {
public:
  std::string getSourceForModuleName(llvm::StringRef ModuleName) {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    auto Iter = ModuleNameToSourceCache.find(ModuleName);
    if (Iter != ModuleNameToSourceCache.end())
      return Iter->second;
    return "";
  }

  void addEntry(llvm::StringRef ModuleName, PathRef Source) {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    ModuleNameToSourceCache[ModuleName] = Source.str();
  }

  void eraseEntry(llvm::StringRef ModuleName) {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    ModuleNameToSourceCache.erase(ModuleName);
  }

private:
  std::mutex CacheMutex;
  llvm::StringMap<std::string> ModuleNameToSourceCache;
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

  std::string getSourceForModuleName(llvm::StringRef ModuleName,
                                     PathRef RequiredSrcFile) override {
    std::string CachedResult = Cache.getSourceForModuleName(ModuleName);

    // Verify Cached Result by seeing if the source declaring the same module
    // as we query.
    if (!CachedResult.empty()) {
      std::string ModuleNameOfCachedSource =
          MDB->getModuleNameForSource(CachedResult);
      if (ModuleNameOfCachedSource == ModuleName)
        return CachedResult;

      // Cached Result is invalid. Clear it.
      Cache.eraseEntry(ModuleName);
    }

    auto Result = MDB->getSourceForModuleName(ModuleName, RequiredSrcFile);
    Cache.addEntry(ModuleName, Result);

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

  ModuleFileCache Cache;
  ModuleNameToSourceCache ProjectModulesCache;
};

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

    if (IsModuleFileUpToDate(ModuleFilePath, BuiltModuleFiles,
                             TFS.view(std::nullopt))) {
      log("Reusing prebuilt module file {0} of module {1} for {2}",
          ModuleFilePath, ModuleName, ModuleUnitFileName);
      BuiltModuleFiles.addModuleFile(
          PrebuiltModuleFile::make(ModuleName, ModuleFilePath));
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

    if (auto Cached = Cache.getModule(ReqModuleName)) {
      if (IsModuleFileUpToDate(Cached->getModuleFilePath(), BuiltModuleFiles,
                               TFS.view(std::nullopt))) {
        log("Reusing module {0} from {1}", ReqModuleName,
            Cached->getModuleFilePath());
        BuiltModuleFiles.addModuleFile(std::move(Cached));
        continue;
      }
      Cache.remove(ReqModuleName);
    }

    std::string ReqFileName =
        MDB.getSourceForModuleName(ReqModuleName, RequiredSource);
    llvm::Expected<std::shared_ptr<BuiltModuleFile>> MF = buildModuleFile(
        ReqModuleName, ReqFileName, getCDB(), TFS, BuiltModuleFiles);
    if (llvm::Error Err = MF.takeError())
      return Err;

    log("Built module {0} to {1}", ReqModuleName, (*MF)->getModuleFilePath());
    Cache.add(ReqModuleName, *MF);
    BuiltModuleFiles.addModuleFile(std::move(*MF));
  }

  return llvm::Error::success();
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
