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
} // namespace

bool ModulesBuilder::buildModuleFile(StringRef ModuleName,
                                     const ThreadsafeFS *TFS,
                                     std::shared_ptr<ProjectModules> MDB,
                                     PathRef ModuleFilesPrefix,
                                     PrerequisiteModules &BuiltModuleFiles) {
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
    if (!buildModuleFile(RequiredModuleName, TFS, MDB, ModuleFilesPrefix,
                         BuiltModuleFiles)) {
      log("Failed to build module {0}", RequiredModuleName);
      return false;
    }
  }

  auto Cmd = CDB.getCompileCommand(ModuleUnitFileName);
  if (!Cmd)
    return false;

  std::string ModuleFileName =
      getUniqueModuleFilePath(ModuleName, ModuleFilesPrefix);
  Cmd->Output = ModuleFileName;

  ParseInputs Inputs;
  Inputs.TFS = TFS;
  Inputs.CompileCommand = std::move(*Cmd);

  IgnoreDiagnostics IgnoreDiags;
  auto CI = buildCompilerInvocation(Inputs, IgnoreDiags);
  if (!CI)
    return false;

  auto FS = Inputs.TFS->view(Inputs.CompileCommand.Directory);
  auto AbsolutePath = getAbsolutePath(Inputs.CompileCommand);
  auto Buf = FS->getBufferForFile(AbsolutePath);
  if (!Buf)
    return false;

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
    return false;

  GenerateModuleInterfaceAction Action;
  Clang->ExecuteAction(Action);

  if (Clang->getDiagnostics().hasErrorOccurred())
    return false;

  BuiltModuleFiles.addModuleFile(ModuleName, ModuleFileName);
  return true;
}

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

  /// We shouldn't add any module files to the FailedPrerequisiteModules.
  void addModuleFile(StringRef ModuleName, StringRef ModuleFilePath) override {
    assert(false && "We shouldn't operator based on failed module files");
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

  bool canReuse(const CompilerInvocation &CI,
                llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>) const override;

  bool isModuleUnitBuilt(StringRef ModuleName) const override {
    constexpr unsigned SmallSizeThreshold = 8;
    if (RequiredModules.size() < SmallSizeThreshold)
      return llvm::any_of(RequiredModules, [&](auto &MF) {
        return MF.ModuleName == ModuleName;
      });

    return BuiltModuleNames.contains(ModuleName);
  }

  void addModuleFile(StringRef ModuleName, StringRef ModuleFilePath) override {
    RequiredModules.emplace_back(ModuleName, ModuleFilePath);
    BuiltModuleNames.insert(ModuleName);
  }

private:
  struct ModuleFile {
    ModuleFile(StringRef ModuleName, PathRef ModuleFilePath)
        : ModuleName(ModuleName.str()), ModuleFilePath(ModuleFilePath.str()) {}

    ModuleFile() = delete;

    ModuleFile(const ModuleFile &) = delete;
    ModuleFile operator=(const ModuleFile &) = delete;

    // The move constructor is needed for llvm::SmallVector.
    ModuleFile(ModuleFile &&Other)
        : ModuleName(std::move(Other.ModuleName)),
          ModuleFilePath(std::move(Other.ModuleFilePath)) {}

    ModuleFile &operator=(ModuleFile &&Other) = delete;

    ~ModuleFile() {
      if (!ModuleFilePath.empty())
        llvm::sys::fs::remove(ModuleFilePath);
    }

    std::string ModuleName;
    std::string ModuleFilePath;
  };

  llvm::SmallVector<ModuleFile, 8> RequiredModules;
  /// A helper class to speedup the query if a module is built.
  llvm::StringSet<> BuiltModuleNames;
};

std::unique_ptr<PrerequisiteModules>
ModulesBuilder::buildPrerequisiteModulesFor(PathRef File,
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
    if (!buildModuleFile(RequiredModuleName, TFS, MDB, ModuleFilesPrefix,
                         *RequiredModules.get())) {
      log("Failed to build module {0}", RequiredModuleName);
      return std::make_unique<FailedPrerequisiteModules>();
    }

  return std::move(RequiredModules);
}

bool StandalonePrerequisiteModules::canReuse(
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
  for (auto &RequiredModule : RequiredModules) {
    StringRef BMIPath = RequiredModule.ModuleFilePath;
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

} // namespace clangd
} // namespace clang
