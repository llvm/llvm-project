//===----------------- ModulesBuilder.cpp ------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ModulesBuilder.h"
#include "PrerequisiteModules.h"
#include "support/Logger.h"

#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"

namespace clang {
namespace clangd {

namespace {

/// Get or create a path to store module files. Generally it should be:
///
///   project_root/.cache/clangd/module_files/{RequiredPrefixDir}/.
///
/// \param MainFile is used to get the root of the project from global
/// compilation database. \param RequiredPrefixDir is used to get the user
/// defined prefix for module files. This is useful when we want to seperate
/// module files. e.g., we want to build module files for the same module unit
/// `a.cppm` with 2 different users `b.cpp` and `c.cpp` and we don't want the
/// module file for `b.cpp` be conflict with the module files for `c.cpp`. Then
/// we can put the 2 module files into different dirs like:
///
///   project_root/.cache/clangd/module_files/b.cpp/a.pcm
///   project_root/.cache/clangd/module_files/c.cpp/a.pcm
llvm::SmallString<256> getModuleFilesPath(PathRef MainFile,
                                          const GlobalCompilationDatabase &CDB,
                                          StringRef RequiredPrefixDir) {
  std::optional<ProjectInfo> PI = CDB.getProjectInfo(MainFile);
  if (!PI)
    return {};

  // FIXME: PI->SourceRoot may be empty, depending on the CDB strategy.
  llvm::SmallString<256> Result(PI->SourceRoot);

  llvm::sys::path::append(Result, ".cache");
  llvm::sys::path::append(Result, "clangd");
  llvm::sys::path::append(Result, "module_files");

  llvm::sys::path::append(Result, RequiredPrefixDir);

  llvm::sys::fs::create_directories(Result, /*IgnoreExisting=*/true);

  return Result;
}

/// Get the absolute path for the filename from the compile command.
llvm::SmallString<128> getAbsolutePath(const tooling::CompileCommand &Cmd) {
  llvm::SmallString<128> AbsolutePath;
  if (llvm::sys::path::is_absolute(Cmd.Filename)) {
    AbsolutePath = Cmd.Filename;
  } else {
    AbsolutePath = Cmd.Directory;
    llvm::sys::path::append(AbsolutePath, Cmd.Filename);
    llvm::sys::path::remove_dots(AbsolutePath, true);
  }
  return AbsolutePath;
}

/// Get a unique module file path under \param ModuleFilesPrefix.
std::string getUniqueModuleFilePath(StringRef ModuleName,
                                    PathRef ModuleFilesPrefix) {
  llvm::SmallString<256> ModuleFilePattern(ModuleFilesPrefix);
  auto [PrimaryModuleName, PartitionName] = ModuleName.split(':');
  llvm::sys::path::append(ModuleFilePattern, PrimaryModuleName);
  if (!PartitionName.empty()) {
    ModuleFilePattern.append("-");
    ModuleFilePattern.append(PartitionName);
  }

  ModuleFilePattern.append("-%%-%%-%%-%%-%%-%%");
  ModuleFilePattern.append(".pcm");

  llvm::SmallString<256> ModuleFilePath;
  llvm::sys::fs::createUniquePath(ModuleFilePattern, ModuleFilePath,
                                  /*MakeAbsolute=*/false);

  return (std::string)ModuleFilePath;
}

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

  ModuleFile &operator=(ModuleFile &&Other) = delete;

  ~ModuleFile() {
    if (!ModuleFilePath.empty())
      llvm::sys::fs::remove(ModuleFilePath);
  }

  std::string ModuleName;
  std::string ModuleFilePath;
};

/// All the required module should be included in BuiltModuleFiles.
std::optional<ModuleFile>
buildModuleFile(StringRef ModuleName, PathRef ModuleUnitFile,
                const GlobalCompilationDatabase &CDB,
                const PrerequisiteModules &BuiltModuleFiles,
                const ThreadsafeFS *TFS, PathRef ModuleFilesPrefix) {
  auto Cmd = CDB.getCompileCommand(ModuleUnitFile);
  if (!Cmd)
    return std::nullopt;

  std::string ModuleFileName =
      getUniqueModuleFilePath(ModuleName, ModuleFilesPrefix);
  Cmd->Output = ModuleFileName;

  std::string CommandLine;
  for (auto &Arg : Cmd->CommandLine)
    CommandLine += Arg + " ";

  ParseInputs Inputs;
  Inputs.TFS = TFS;
  Inputs.CompileCommand = std::move(*Cmd);

  IgnoreDiagnostics IgnoreDiags;
  auto CI = buildCompilerInvocation(Inputs, IgnoreDiags);
  if (!CI) {
    log("Failed to build module {0} since build Compiler invocation failed", ModuleName);
    return std::nullopt;
  }

  auto FS = Inputs.TFS->view(Inputs.CompileCommand.Directory);
  auto AbsolutePath = getAbsolutePath(Inputs.CompileCommand);
  auto Buf = FS->getBufferForFile(AbsolutePath);
  if (!Buf) {
    log("Failed to build module {0} since get buffer failed", ModuleName);
    return std::nullopt;
  }
  
  // Try to use the built module files from clangd first.
  BuiltModuleFiles.adjustHeaderSearchOptions(CI->getHeaderSearchOpts());

  // Hash the contents of input files and store the hash value to the BMI files.
  // So that we can check if the files are still valid when we want to reuse the
  // BMI files.
  CI->getHeaderSearchOpts().ValidateASTInputFilesContent = true;

  BuiltModuleFiles.adjustHeaderSearchOptions(CI->getHeaderSearchOpts());

  CI->getFrontendOpts().OutputFile = Inputs.CompileCommand.Output;
  auto Clang =
      prepareCompilerInstance(std::move(CI), /*Preamble=*/nullptr,
                              std::move(*Buf), std::move(FS), IgnoreDiags);
  if (!Clang) {
    log("Failed to build module {0} since build compiler instance failed", ModuleName);
    return std::nullopt;
  }

  GenerateModuleInterfaceAction Action;
  Clang->ExecuteAction(Action);

  if (Clang->getDiagnostics().hasErrorOccurred()) {
    log("Failed to build module {0} since error occurred failed", ModuleName);
    log("Failing Command line {0}", CommandLine);
    return std::nullopt;
  }

  return ModuleFile{ModuleName, ModuleFileName};
}
} // namespace

/// FailedPrerequisiteModules - stands for the PrerequisiteModules which has
/// errors happened during the building process.
class FailedPrerequisiteModules : public PrerequisiteModules {
public:
  ~FailedPrerequisiteModules() override = default;

  /// We shouldn't adjust the compilation commands based on
  /// FailedPrerequisiteModules.
  void adjustHeaderSearchOptions(HeaderSearchOptions &Options) const override {
  }
  /// FailedPrerequisiteModules can never be reused.
  bool
  canReuse(const CompilerInvocation &CI,
           llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>) const override {
    return false;
  }

  /// No module unit got built in FailedPrerequisiteModules.
  bool isModuleUnitBuilt(StringRef ModuleName) const override { return false; }

  // FailedPrerequisiteModules don't require any module file.
  void getRequiredModuleFiles(
      llvm::SmallVector<StringRef> &ModuleFiles) const override {
    return;
  }
};

/// StandalonePrerequisiteModules - stands for PrerequisiteModules for which all
/// the required modules are built successfully. All the module files
/// are owned by the StandalonePrerequisiteModules class.
///
/// All the built module files won't be shared with other instances of the
/// class. So that we can avoid worrying thread safety.
///
/// We don't need to worry about duplicated module names here since the standard
/// guarantees the module names should be unique to a program.
class StandalonePrerequisiteModules : public PrerequisiteModules {
public:
  StandalonePrerequisiteModules() = default;

  StandalonePrerequisiteModules(const StandalonePrerequisiteModules &) = delete;
  StandalonePrerequisiteModules
  operator=(const StandalonePrerequisiteModules &) = delete;
  StandalonePrerequisiteModules(StandalonePrerequisiteModules &&) = delete;
  StandalonePrerequisiteModules
  operator=(StandalonePrerequisiteModules &&) = delete;

  ~StandalonePrerequisiteModules() override = default;

  void adjustHeaderSearchOptions(HeaderSearchOptions &Options) const override {
    // Appending all built module files.
    for (auto &RequiredModule : RequiredModules)
      Options.PrebuiltModuleFiles.insert_or_assign(
          RequiredModule.ModuleName, RequiredModule.ModuleFilePath);
  }

  bool isModuleUnitBuilt(StringRef ModuleName) const override {
    constexpr unsigned SmallSizeThreshold = 8;
    if (RequiredModules.size() < SmallSizeThreshold)
      return llvm::any_of(RequiredModules, [&](auto &MF) {
        return MF.ModuleName == ModuleName;
      });

    return BuiltModuleNames.contains(ModuleName);
  }

  void addModuleFile(ModuleFile MF) {
    BuiltModuleNames.insert(MF.ModuleName);
    RequiredModules.emplace_back(std::move(MF));
  }

private:
  void getRequiredModuleFiles(
      llvm::SmallVector<StringRef> &ModuleFiles) const override {
    for (auto &MF : RequiredModules)
      ModuleFiles.push_back(MF.ModuleFilePath);
  }

  llvm::SmallVector<ModuleFile, 8> RequiredModules;
  /// A helper class to speedup the query if a module is built.
  llvm::StringSet<> BuiltModuleNames;
};

bool PrerequisiteModules::canReuse(
    const CompilerInvocation &CI,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) const {
  CompilerInstance Clang;

  Clang.setInvocation(std::make_shared<CompilerInvocation>(CI));
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(new DiagnosticOptions());
  Clang.setDiagnostics(Diags.get());

  FileManager *FM = Clang.createFileManager(VFS);
  Clang.createSourceManager(*FM);

  if (!Clang.createTarget())
    return false;

  assert(Clang.getHeaderSearchOptsPtr());
  adjustHeaderSearchOptions(Clang.getHeaderSearchOpts());
  // Since we don't need to compile the source code actually, the TU kind here
  // doesn't matter.
  Clang.createPreprocessor(TU_Complete);
  Clang.getHeaderSearchOpts().ForceCheckCXX20ModulesInputFiles = true;
  Clang.getHeaderSearchOpts().ValidateASTInputFilesContent = true;

  Clang.createASTReader();
  SmallVector<StringRef> BMIPaths;
  getRequiredModuleFiles(BMIPaths);
  for (StringRef BMIPath : BMIPaths) {
    auto ReadResult =
        Clang.getASTReader()->ReadAST(BMIPath, serialization::MK_MainFile,
                                      SourceLocation(), ASTReader::ARR_None);

    if (ReadResult != ASTReader::Success) {
      log("Failed to reuse {0}", BMIPath);
      return false;
    }
  }

  return true;
}

class StandaloneModulesBuilder : public ModulesBuilder {
public:
  StandaloneModulesBuilder() = delete;

  StandaloneModulesBuilder(const GlobalCompilationDatabase &CDB) : CDB(CDB) {}

  StandaloneModulesBuilder(const StandaloneModulesBuilder &) = delete;
  StandaloneModulesBuilder(StandaloneModulesBuilder &&) = delete;

  StandaloneModulesBuilder &
  operator=(const StandaloneModulesBuilder &) = delete;
  StandaloneModulesBuilder &operator=(StandaloneModulesBuilder &&) = delete;

  std::unique_ptr<PrerequisiteModules>
  buildPrerequisiteModulesFor(PathRef File, const ThreadsafeFS *TFS) override;

private:
  bool getOrBuildModuleFile(StringRef ModuleName, const ThreadsafeFS *TFS,
                            std::shared_ptr<ProjectModules> MDB,
                            PathRef ModuleFilesPrefix,
                            StandalonePrerequisiteModules &RequiredModules);

  const GlobalCompilationDatabase &CDB;
};

std::unique_ptr<PrerequisiteModules>
StandaloneModulesBuilder::buildPrerequisiteModulesFor(PathRef File,
                                                      const ThreadsafeFS *TFS) {
  std::shared_ptr<ProjectModules> MDB = CDB.getProjectModules(File);
  if (!MDB)
    return {};

  std::vector<std::string> RequiredModuleNames = MDB->getRequiredModules(File);
  if (RequiredModuleNames.empty())
    return {};

  llvm::SmallString<256> ModuleFilesPrefix =
      getModuleFilesPath(File, CDB, llvm::sys::path::filename(File));

  auto RequiredModules = std::make_unique<StandalonePrerequisiteModules>();

  for (const std::string &RequiredModuleName : RequiredModuleNames)
    // Return early if there is any error.
    if (!getOrBuildModuleFile(RequiredModuleName, TFS, MDB, ModuleFilesPrefix,
                              *RequiredModules.get())) {
      log("Failed to build module {0}", RequiredModuleName);
      return std::make_unique<FailedPrerequisiteModules>();
    }

  return std::move(RequiredModules);
}

bool StandaloneModulesBuilder::getOrBuildModuleFile(
    StringRef ModuleName, const ThreadsafeFS *TFS,
    std::shared_ptr<ProjectModules> MDB, PathRef ModuleFilesPrefix,
    StandalonePrerequisiteModules &BuiltModuleFiles) {
  if (BuiltModuleFiles.isModuleUnitBuilt(ModuleName))
    return true;

  PathRef ModuleUnitFileName = MDB->getSourceForModuleName(ModuleName);
  /// It is possible that we're meeting third party modules (modules whose
  /// source are not in the project. e.g, the std module may be a third-party
  /// module for most project) or something wrong with the implementation of
  /// ProjectModules.
  /// FIXME: How should we treat third party modules here? If we want to ignore
  /// third party modules, we should return true instead of false here.
  /// Currently we simply bail out.
  if (ModuleUnitFileName.empty())
    return false;

  for (auto &RequiredModuleName : MDB->getRequiredModules(ModuleUnitFileName)) {
    // Return early if there are errors building the module file.
    if (!getOrBuildModuleFile(RequiredModuleName, TFS, MDB, ModuleFilesPrefix,
                              BuiltModuleFiles)) {
      log("Failed to build module {0}", RequiredModuleName);
      return false;
    }
  }

  std::optional<ModuleFile> MF =
      buildModuleFile(ModuleName, ModuleUnitFileName, CDB, BuiltModuleFiles,
                      TFS, ModuleFilesPrefix);
  if (!MF)
    return false;

  BuiltModuleFiles.addModuleFile(std::move(*MF));
  return true;
}

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

  bool isModuleUnitBuilt(StringRef ModuleName) const override {
    constexpr unsigned SmallSizeThreshold = 8;
    if (RequiredModules.size() < SmallSizeThreshold)
      return llvm::any_of(RequiredModules, [&](auto &MF) {
        return MF->ModuleName == ModuleName;
      });

    return BuiltModuleNames.contains(ModuleName);
  }

  void addModuleFile(std::shared_ptr<ModuleFile> BMI) {
    BuiltModuleNames.insert(BMI->ModuleName);
    RequiredModules.emplace_back(std::move(BMI));
  }

private:
  void getRequiredModuleFiles(
      llvm::SmallVector<StringRef> &ModuleFiles) const override {
    for (auto &MF : RequiredModules)
      ModuleFiles.push_back(MF->ModuleFilePath);
  }

  llvm::SmallVector<std::shared_ptr<ModuleFile>, 8> RequiredModules;
  /// A helper class to speedup the query if a module is built.
  llvm::StringSet<> BuiltModuleNames;
};

class ReusableModulesBuilder : public ModulesBuilder {
public:
  ReusableModulesBuilder() = delete;

  ReusableModulesBuilder(const GlobalCompilationDatabase &CDB) : CDB(CDB) {}

  ReusableModulesBuilder(const ReusableModulesBuilder &) = delete;
  ReusableModulesBuilder(ReusableModulesBuilder &&) = delete;

  ReusableModulesBuilder &operator=(const ReusableModulesBuilder &) = delete;
  ReusableModulesBuilder &operator=(ReusableModulesBuilder &&) = delete;

  std::unique_ptr<PrerequisiteModules>
  buildPrerequisiteModulesFor(PathRef File, const ThreadsafeFS *TFS) override;

private:
  bool getOrBuildModuleFile(StringRef ModuleName, const ThreadsafeFS *TFS,
                            std::shared_ptr<ProjectModules> MDB,
                            ReusablePrerequisiteModules &RequiredModules);

  std::shared_ptr<ModuleFile>
  getValidModuleFile(StringRef ModuleName,
                     std::shared_ptr<ProjectModules> &MDB);
  /// This should only be called by getValidModuleFile. This is unlocked version
  /// of getValidModuleFile. This is extracted to avoid dead locks when
  /// recursing.
  std::shared_ptr<ModuleFile>
  isValidModuleFileUnlocked(StringRef ModuleName,
                            std::shared_ptr<ProjectModules> &MDB);
  void maintainModuleFiles(StringRef ModuleName);

  llvm::StringMap<std::weak_ptr<ModuleFile>> ModuleFiles;
  std::mutex ModuleFilesMutex;

  // We should only build a unique module at most at the same time.
  // When we want to build a module
  llvm::StringMap<std::shared_ptr<std::mutex>> BuildingModuleMutexes;
  llvm::StringMap<std::shared_ptr<std::condition_variable>> BuildingModuleCVs;
  // Lock when we accessing ModuleBuildingCVs and ModuleBuildingMutexes.
  std::mutex ModulesBuildingMutex;

  struct ModuleBuildingSharedOwner {
  public:
    ModuleBuildingSharedOwner(StringRef ModuleName,
                              std::shared_ptr<std::mutex> &Mutex,
                              std::shared_ptr<std::condition_variable> &CV,
                              ReusableModulesBuilder &Builder)
        : ModuleName(ModuleName), Mutex(Mutex), CV(CV), Builder(Builder) {
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
    ReusableModulesBuilder &Builder;
    bool IsFirstTask;
  };

  ModuleBuildingSharedOwner
  getOrCreateModuleBuildingCVAndLock(StringRef ModuleName);

  const GlobalCompilationDatabase &CDB;
};

ReusableModulesBuilder::ModuleBuildingSharedOwner::
    ~ModuleBuildingSharedOwner() {
  std::lock_guard<std::mutex> _(Builder.ModulesBuildingMutex);

  Mutex.reset();
  CV.reset();

  // Try to release the memory in builder if possible.
  if (auto Iter = Builder.BuildingModuleCVs.find(ModuleName);
      Iter != Builder.BuildingModuleCVs.end() &&
      Iter->getValue().use_count() == 1)
    Builder.BuildingModuleCVs.erase(Iter);

  if (auto Iter = Builder.BuildingModuleMutexes.find(ModuleName);
      Iter != Builder.BuildingModuleMutexes.end() &&
      Iter->getValue().use_count() == 1)
    Builder.BuildingModuleMutexes.erase(Iter);
}

std::shared_ptr<ModuleFile> ReusableModulesBuilder::isValidModuleFileUnlocked(
    StringRef ModuleName, std::shared_ptr<ProjectModules> &MDB) {
  auto Iter = ModuleFiles.find(ModuleName);
  if (Iter != ModuleFiles.end()) {
    if (Iter->second.expired() ||
        llvm::any_of(
            MDB->getRequiredModules(MDB->getSourceForModuleName(ModuleName)),
            [&MDB, this](auto &&RequiredModuleName) {
              return !isValidModuleFileUnlocked(RequiredModuleName, MDB);
            })) {
      ModuleFiles.erase(Iter);
      return nullptr;
    }

    return Iter->second.lock();
  }

  return nullptr;
}

std::shared_ptr<ModuleFile> ReusableModulesBuilder::getValidModuleFile(
    StringRef ModuleName, std::shared_ptr<ProjectModules> &MDB) {
  std::lock_guard<std::mutex> _(ModuleFilesMutex);

  return isValidModuleFileUnlocked(ModuleName, MDB);
}

void ReusableModulesBuilder::maintainModuleFiles(StringRef ModuleName) {
  std::lock_guard<std::mutex> _(ModuleFilesMutex);

  auto Iter = ModuleFiles.find(ModuleName);
  if (Iter != ModuleFiles.end())
    ModuleFiles.erase(Iter);
}

std::unique_ptr<PrerequisiteModules>
ReusableModulesBuilder::buildPrerequisiteModulesFor(PathRef File,
                                                    const ThreadsafeFS *TFS) {
  std::shared_ptr<ProjectModules> MDB = CDB.getProjectModules(File);
  if (!MDB)
    return {};

  std::optional<std::string> ModuleName = MDB->getModuleName(File);
  if (ModuleName)
    maintainModuleFiles(*ModuleName);

  std::vector<std::string> RequiredModuleNames = MDB->getRequiredModules(File);
  if (RequiredModuleNames.empty())
    return {};

  auto RequiredModules = std::make_unique<ReusablePrerequisiteModules>();

  for (const std::string &RequiredModuleName : RequiredModuleNames)
    // Return early if there is any error.
    if (!getOrBuildModuleFile(RequiredModuleName, TFS, MDB,
                              *RequiredModules.get())) {
      log("Failed to build module {0}", RequiredModuleName);
      return std::make_unique<FailedPrerequisiteModules>();
    }

  return std::move(RequiredModules);
}

ReusableModulesBuilder::ModuleBuildingSharedOwner
ReusableModulesBuilder::getOrCreateModuleBuildingCVAndLock(
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

bool ReusableModulesBuilder::getOrBuildModuleFile(
    StringRef ModuleName, const ThreadsafeFS *TFS,
    std::shared_ptr<ProjectModules> MDB,
    ReusablePrerequisiteModules &BuiltModuleFiles) {
  if (BuiltModuleFiles.isModuleUnitBuilt(ModuleName))
    return true;

  PathRef ModuleUnitFileName = MDB->getSourceForModuleName(ModuleName);
  /// It is possible that we're meeting third party modules (modules whose
  /// source are not in the project. e.g, the std module may be a third-party
  /// module for most project) or something wrong with the implementation of
  /// ProjectModules.
  /// FIXME: How should we treat third party modules here? If we want to ignore
  /// third party modules, we should return true instead of false here.
  /// Currently we simply bail out.
  if (ModuleUnitFileName.empty())
    return false;

  if (std::shared_ptr<ModuleFile> Cached =
          getValidModuleFile(ModuleName, MDB)) {
    log("Reusing Built Module {0} with {1}", Cached->ModuleName, Cached->ModuleFilePath);
    BuiltModuleFiles.addModuleFile(Cached);
    return true;
  }

  for (auto &RequiredModuleName : MDB->getRequiredModules(ModuleUnitFileName)) {
    // Return early if there are errors building the module file.
    if (!getOrBuildModuleFile(RequiredModuleName, TFS, MDB, BuiltModuleFiles)) {
      log("Failed to build module {0}", RequiredModuleName);
      return false;
    }
  }

  ModuleBuildingSharedOwner ModuleBuildingOwner =
      getOrCreateModuleBuildingCVAndLock(ModuleName);

  std::condition_variable &CV = ModuleBuildingOwner.getCV();
  std::unique_lock lk(ModuleBuildingOwner.getMutex());
  if (!ModuleBuildingOwner.isUniqueBuildingOwner()) {
    CV.wait(lk);

    // Try to access the built module files from other threads manually.
    // We don't call getValidModuleFile here since it may be too heavy.
    std::lock_guard<std::mutex> _(ModuleFilesMutex);
    auto Iter = ModuleFiles.find(ModuleName);
    if (Iter != ModuleFiles.end()) {
      BuiltModuleFiles.addModuleFile(Iter->second.lock());
      return true;
    }
  }

  llvm::SmallString<256> ModuleFilesPrefix = getModuleFilesPath(
      ModuleUnitFileName, CDB, llvm::sys::path::filename(ModuleUnitFileName));

  std::optional<ModuleFile> MF =
      buildModuleFile(ModuleName, ModuleUnitFileName, CDB, BuiltModuleFiles,
                      TFS, ModuleFilesPrefix);
  bool BuiltSuccessed = (bool)MF;
  if (MF) {
    std::lock_guard<std::mutex> _(ModuleFilesMutex);
    auto BuiltModuleFile = std::make_shared<ModuleFile>(std::move(*MF));
    ModuleFiles.insert_or_assign(ModuleName, BuiltModuleFile);
    BuiltModuleFiles.addModuleFile(std::move(BuiltModuleFile));
  }

  CV.notify_all();
  return BuiltSuccessed;
}

std::unique_ptr<ModulesBuilder>
ModulesBuilder::create(ModulesBuilderKind Kind,
                       const GlobalCompilationDatabase &CDB) {
  switch (Kind) {
  case ModulesBuilderKind::StandaloneModulesBuilder:
    return std::make_unique<StandaloneModulesBuilder>(CDB);
  case ModulesBuilderKind::ReusableModulesBuilder:
    return std::make_unique<ReusableModulesBuilder>(CDB);
  }
  llvm_unreachable("Unknown Modules Build Kind.");
}

} // namespace clangd
} // namespace clang
