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
#include "clang/Serialization/InMemoryModuleCache.h"
#include "llvm/ADT/ScopeExit.h"

namespace clang {
namespace clangd {

namespace {

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

struct ModuleFile {
  ModuleFile(StringRef ModuleName, PathRef ModuleFilePath)
      : ModuleName(ModuleName.str()), ModuleFilePath(ModuleFilePath.str()) {}

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

  ~ModuleFile() {
    if (!ModuleFilePath.empty())
      llvm::sys::fs::remove(ModuleFilePath);
  }

  std::string ModuleName;
  std::string ModuleFilePath;
};

bool IsModuleFileUpToDate(PathRef ModuleFilePath,
                          const PrerequisiteModules &RequisiteModules,
                          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) {
  auto HSOpts = std::make_shared<HeaderSearchOptions>();
  RequisiteModules.adjustHeaderSearchOptions(*HSOpts);
  HSOpts->ForceCheckCXX20ModulesInputFiles = true;
  HSOpts->ValidateASTInputFilesContent = true;

  clang::clangd::IgnoreDiagnostics IgnoreDiags;
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(new DiagnosticOptions, &IgnoreDiags,
                                          /*ShouldOwnClient=*/false);

  LangOptions LangOpts;
  LangOpts.SkipODRCheckInGMF = true;

  FileManager FileMgr(FileSystemOptions(), VFS);

  SourceManager SourceMgr(*Diags, FileMgr);

  HeaderSearch HeaderInfo(HSOpts, SourceMgr, *Diags, LangOpts,
                          /*Target=*/nullptr);

  TrivialModuleLoader ModuleLoader;
  Preprocessor PP(std::make_shared<PreprocessorOptions>(), *Diags, LangOpts,
                  SourceMgr, HeaderInfo, ModuleLoader);

  IntrusiveRefCntPtr<InMemoryModuleCache> ModuleCache = new InMemoryModuleCache;
  PCHContainerOperations PCHOperations;
  ASTReader Reader(PP, *ModuleCache, /*ASTContext=*/nullptr,
                   PCHOperations.getRawReader(), {});

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

// ReusablePrerequisiteModules - stands for PrerequisiteModules for which all
// the required modules are built successfully. All the module files
// are owned by the modules builder.
class ReusablePrerequisiteModules : public PrerequisiteModules {
public:
  ReusablePrerequisiteModules() = default;

  ReusablePrerequisiteModules(const ReusablePrerequisiteModules &) = delete;
  ReusablePrerequisiteModules
  operator=(const ReusablePrerequisiteModules &) = delete;
  ReusablePrerequisiteModules(ReusablePrerequisiteModules &&) = delete;
  ReusablePrerequisiteModules
  operator=(ReusablePrerequisiteModules &&) = delete;

  ~ReusablePrerequisiteModules() override = default;

  void adjustHeaderSearchOptions(HeaderSearchOptions &Options) const override {
    // Appending all built module files.
    for (auto &RequiredModule : RequiredModules)
      Options.PrebuiltModuleFiles.insert_or_assign(
          RequiredModule->ModuleName, RequiredModule->ModuleFilePath);
  }

  bool canReuse(const CompilerInvocation &CI,
                llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>) const override;

  bool isModuleUnitBuilt(llvm::StringRef ModuleName) const {
    return BuiltModuleNames.contains(ModuleName);
  }

  void addModuleFile(std::shared_ptr<ModuleFile> BMI) {
    BuiltModuleNames.insert(BMI->ModuleName);
    RequiredModules.emplace_back(std::move(BMI));
  }

private:
  mutable llvm::SmallVector<std::shared_ptr<ModuleFile>, 8> RequiredModules;
  // A helper class to speedup the query if a module is built.
  llvm::StringSet<> BuiltModuleNames;
};

/// Build a module file for module with `ModuleName`. The information of built
/// module file are stored in \param BuiltModuleFiles.
llvm::Expected<ModuleFile>
buildModuleFile(llvm::StringRef ModuleName, PathRef ModuleUnitFileName,
                const GlobalCompilationDatabase &CDB, const ThreadsafeFS &TFS,
                PathRef ModuleFilesPrefix,
                const ReusablePrerequisiteModules &BuiltModuleFiles) {
  // Try cheap operation earlier to boil-out cheaply if there are problems.
  auto Cmd = CDB.getCompileCommand(ModuleUnitFileName);
  if (!Cmd)
    return llvm::createStringError(
        llvm::formatv("No compile command for {0}", ModuleUnitFileName));

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

  if (Clang->getDiagnostics().hasErrorOccurred())
    return llvm::createStringError("Compilation failed");

  return ModuleFile{ModuleName, Inputs.CompileCommand.Output};
}

bool ReusablePrerequisiteModules::canReuse(
    const CompilerInvocation &CI,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) const {
  if (RequiredModules.empty())
    return true;

  SmallVector<StringRef> BMIPaths;
  for (auto &MF : RequiredModules)
    BMIPaths.push_back(MF->ModuleFilePath);
  return IsModuleFilesUpToDate(BMIPaths, *this, VFS);
}
} // namespace

class ModulesBuilder::ModuleFileCache {
public:
  ModuleFileCache(const GlobalCompilationDatabase &CDB) : CDB(CDB) {}

  llvm::Error
  getOrBuildModuleFile(StringRef ModuleName, const ThreadsafeFS &TFS,
                       ProjectModules &MDB,
                       ReusablePrerequisiteModules &RequiredModules);
  const GlobalCompilationDatabase &getCDB() const { return CDB; }

private:
  std::shared_ptr<ModuleFile>
  getValidModuleFile(StringRef ModuleName, ProjectModules &MDB,
                     const ThreadsafeFS &TFS,
                     PrerequisiteModules &BuiltModuleFiles);

  /// This should only be called by getValidModuleFile. This is unlocked version
  /// of getValidModuleFile. The function is extracted to avoid dead locks when
  /// recursing.
  std::shared_ptr<ModuleFile>
  isValidModuleFileUnlocked(StringRef ModuleName, ProjectModules &MDB,
                            const ThreadsafeFS &TFS,
                            PrerequisiteModules &BuiltModuleFiles);

  void startBuildingModule(StringRef ModuleName) {
    std::lock_guard<std::mutex> _(ModulesBuildingMutex);
    BuildingModules.insert(ModuleName);
  }
  void endBuildingModule(StringRef ModuleName) {
    std::lock_guard<std::mutex> _(ModulesBuildingMutex);
    BuildingModules.erase(ModuleName);
  }
  bool isBuildingModule(StringRef ModuleName) {
    std::lock_guard<std::mutex> _(ModulesBuildingMutex);
    return BuildingModules.contains(ModuleName);
  }

  const GlobalCompilationDatabase &CDB;

  llvm::StringMap<std::shared_ptr<ModuleFile>> ModuleFiles;
  // Mutex to guard accesses to ModuleFiles.
  std::mutex ModuleFilesMutex;

  // We should only build a unique module at most at the same time.
  // When we want to build a module, use the mutex to lock it and use the
  // condition variable to notify other threads the status of the build results.
  //
  // Store the mutex and the condition_variable in shared_ptr since they may be
  // accessed by many threads.
  llvm::StringMap<std::shared_ptr<std::mutex>> BuildingModuleMutexes;
  llvm::StringMap<std::shared_ptr<std::condition_variable>> BuildingModuleCVs;
  // The building modules set. A successed built module or a failed module or
  // an unbuilt module shouldn't be in this set.
  // This set is helpful to control the behavior of the condition variables.
  llvm::StringSet<> BuildingModules;
  // Lock when we access BuildingModules, BuildingModuleMutexes and
  // BuildingModuleCVs.
  std::mutex ModulesBuildingMutex;

  /// An RAII object to guard the process to build a specific module.
  struct ModuleBuildingSharedOwner {
  public:
    ModuleBuildingSharedOwner(StringRef ModuleName,
                              std::shared_ptr<std::mutex> &Mutex,
                              std::shared_ptr<std::condition_variable> &CV,
                              ModuleFileCache &Cache)
        : ModuleName(ModuleName), Mutex(Mutex), CV(CV), Cache(Cache) {
      IsFirstTask = (Mutex.use_count() == 2);
    }

    ~ModuleBuildingSharedOwner();

    bool isUniqueBuildingOwner() { return IsFirstTask; }

    std::mutex &getMutex() { return *Mutex; }

    std::condition_variable &getCV() { return *CV; }

  private:
    StringRef ModuleName;
    std::shared_ptr<std::mutex> Mutex;
    std::shared_ptr<std::condition_variable> CV;
    ModuleFileCache &Cache;
    bool IsFirstTask;
  };

  ModuleBuildingSharedOwner
  getOrCreateModuleBuildingOwner(StringRef ModuleName);
};

ModulesBuilder::ModuleFileCache::ModuleBuildingSharedOwner::
    ~ModuleBuildingSharedOwner() {
  std::lock_guard<std::mutex> _(Cache.ModulesBuildingMutex);

  Mutex.reset();
  CV.reset();

  // Try to release the memory in builder if possible.
  if (auto Iter = Cache.BuildingModuleCVs.find(ModuleName);
      Iter != Cache.BuildingModuleCVs.end() &&
      Iter->getValue().use_count() == 1)
    Cache.BuildingModuleCVs.erase(Iter);

  if (auto Iter = Cache.BuildingModuleMutexes.find(ModuleName);
      Iter != Cache.BuildingModuleMutexes.end() &&
      Iter->getValue().use_count() == 1)
    Cache.BuildingModuleMutexes.erase(Iter);
}

std::shared_ptr<ModuleFile>
ModulesBuilder::ModuleFileCache::isValidModuleFileUnlocked(
    StringRef ModuleName, ProjectModules &MDB, const ThreadsafeFS &TFS,
    PrerequisiteModules &BuiltModuleFiles) {
  auto Iter = ModuleFiles.find(ModuleName);
  if (Iter != ModuleFiles.end()) {
    if (!IsModuleFileUpToDate(Iter->second->ModuleFilePath, BuiltModuleFiles,
                              TFS.view(std::nullopt))) {
      log("Found not-up-date module file {0} for module {1} in cache",
          Iter->second->ModuleFilePath, ModuleName);
      ModuleFiles.erase(Iter);
      return nullptr;
    }

    if (llvm::any_of(
            MDB.getRequiredModules(MDB.getSourceForModuleName(ModuleName)),
            [&MDB, &TFS, &BuiltModuleFiles, this](auto &&RequiredModuleName) {
              return !isValidModuleFileUnlocked(RequiredModuleName, MDB, TFS,
                                                BuiltModuleFiles);
            })) {
      ModuleFiles.erase(Iter);
      return nullptr;
    }

    return Iter->second;
  }

  log("Don't find {0} in cache", ModuleName);

  return nullptr;
}

std::shared_ptr<ModuleFile> ModulesBuilder::ModuleFileCache::getValidModuleFile(
    StringRef ModuleName, ProjectModules &MDB, const ThreadsafeFS &TFS,
    PrerequisiteModules &BuiltModuleFiles) {
  std::lock_guard<std::mutex> _(ModuleFilesMutex);

  return isValidModuleFileUnlocked(ModuleName, MDB, TFS, BuiltModuleFiles);
}

ModulesBuilder::ModuleFileCache::ModuleBuildingSharedOwner
ModulesBuilder::ModuleFileCache::getOrCreateModuleBuildingOwner(
    StringRef ModuleName) {
  std::lock_guard<std::mutex> _(ModulesBuildingMutex);

  auto MutexIter = BuildingModuleMutexes.find(ModuleName);
  if (MutexIter == BuildingModuleMutexes.end())
    MutexIter = BuildingModuleMutexes
                    .try_emplace(ModuleName, std::make_shared<std::mutex>())
                    .first;

  auto CVIter = BuildingModuleCVs.find(ModuleName);
  if (CVIter == BuildingModuleCVs.end())
    CVIter = BuildingModuleCVs
                 .try_emplace(ModuleName,
                              std::make_shared<std::condition_variable>())
                 .first;

  return ModuleBuildingSharedOwner(ModuleName, MutexIter->getValue(),
                                   CVIter->getValue(), *this);
}

llvm::Error ModulesBuilder::ModuleFileCache::getOrBuildModuleFile(
    StringRef ModuleName, const ThreadsafeFS &TFS, ProjectModules &MDB,
    ReusablePrerequisiteModules &BuiltModuleFiles) {
  if (BuiltModuleFiles.isModuleUnitBuilt(ModuleName))
    return llvm::Error::success();

  PathRef ModuleUnitFileName = MDB.getSourceForModuleName(ModuleName);
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

  for (auto &RequiredModuleName : MDB.getRequiredModules(ModuleUnitFileName))
    // Return early if there are errors building the module file.
    if (!getOrBuildModuleFile(RequiredModuleName, TFS, MDB, BuiltModuleFiles))
      return llvm::createStringError(
          llvm::formatv("Failed to build module {0}", RequiredModuleName));

  if (std::shared_ptr<ModuleFile> Cached =
          getValidModuleFile(ModuleName, MDB, TFS, BuiltModuleFiles)) {
    log("Reusing module {0} from {1}", ModuleName, Cached->ModuleFilePath);
    BuiltModuleFiles.addModuleFile(Cached);
    return llvm::Error::success();
  }

  ModuleBuildingSharedOwner ModuleBuildingOwner =
      getOrCreateModuleBuildingOwner(ModuleName);

  std::condition_variable &CV = ModuleBuildingOwner.getCV();
  std::unique_lock<std::mutex> lk(ModuleBuildingOwner.getMutex());
  if (!ModuleBuildingOwner.isUniqueBuildingOwner()) {
    log("Waiting other task for module {0}", ModuleName);
    CV.wait(lk, [this, ModuleName] { return !isBuildingModule(ModuleName); });

    // Try to access the built module files from other threads manually.
    // We don't call getValidModuleFile here since it may be too heavy.
    std::lock_guard<std::mutex> _(ModuleFilesMutex);
    auto Iter = ModuleFiles.find(ModuleName);
    if (Iter != ModuleFiles.end()) {
      log("Got module file from other task building {0}", ModuleName);
      BuiltModuleFiles.addModuleFile(Iter->second);
      return llvm::Error::success();
    }

    // If the module file is not in the cache, it indicates that the building
    // from other thread failed, so we give up earlier in this case to avoid
    // wasting time.
    return llvm::createStringError(llvm::formatv(
        "The module file {0} may be failed to build in other thread.",
        ModuleName));
  }

  log("Building module {0}", ModuleName);
  startBuildingModule(ModuleName);

  auto _ = llvm::make_scope_exit([&]() {
    endBuildingModule(ModuleName);
    CV.notify_all();
  });

  llvm::SmallString<256> ModuleFilesPrefix =
      getUniqueModuleFilesPath(ModuleUnitFileName);

  llvm::Expected<ModuleFile> MF =
      buildModuleFile(ModuleName, ModuleUnitFileName, CDB, TFS,
                      ModuleFilesPrefix, BuiltModuleFiles);
  if (llvm::Error Err = MF.takeError())
    return Err;

  log("Built module {0} to {1}", ModuleName, MF->ModuleFilePath);

  std::lock_guard<std::mutex> __(ModuleFilesMutex);
  auto BuiltModuleFile = std::make_shared<ModuleFile>(std::move(*MF));
  ModuleFiles.insert_or_assign(ModuleName, BuiltModuleFile);
  BuiltModuleFiles.addModuleFile(std::move(BuiltModuleFile));

  return llvm::Error::success();
}
std::unique_ptr<PrerequisiteModules>
ModulesBuilder::buildPrerequisiteModulesFor(PathRef File,
                                            const ThreadsafeFS &TFS) {
  std::unique_ptr<ProjectModules> MDB =
      MFCache->getCDB().getProjectModules(File);
  if (!MDB) {
    elog("Failed to get Project Modules information for {0}", File);
    return std::make_unique<FailedPrerequisiteModules>();
  }

  std::vector<std::string> RequiredModuleNames = MDB->getRequiredModules(File);
  if (RequiredModuleNames.empty())
    return std::make_unique<ReusablePrerequisiteModules>();

  llvm::SmallString<256> ModuleFilesPrefix = getUniqueModuleFilesPath(File);

  log("Trying to build required modules for {0} in {1}", File,
      ModuleFilesPrefix);

  auto RequiredModules = std::make_unique<ReusablePrerequisiteModules>();

  for (llvm::StringRef RequiredModuleName : RequiredModuleNames) {
    // Return early if there is any error.
    if (llvm::Error Err = MFCache->getOrBuildModuleFile(
            RequiredModuleName, TFS, *MDB.get(), *RequiredModules.get())) {
      elog("Failed to build module {0}; due to {1}", RequiredModuleName,
           toString(std::move(Err)));
      return std::make_unique<FailedPrerequisiteModules>();
    }
  }

  log("Built required modules for {0} in {1}", File, ModuleFilesPrefix);

  return std::move(RequiredModules);
}

ModulesBuilder::ModulesBuilder(const GlobalCompilationDatabase &CDB) {
  MFCache = std::make_unique<ModuleFileCache>(CDB);
}

ModulesBuilder::~ModulesBuilder() {}

} // namespace clangd
} // namespace clang
