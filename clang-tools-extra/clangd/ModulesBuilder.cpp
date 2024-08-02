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

// StandalonePrerequisiteModules - stands for PrerequisiteModules for which all
// the required modules are built successfully. All the module files
// are owned by the StandalonePrerequisiteModules class.
//
// Any of the built module files won't be shared with other instances of the
// class. So that we can avoid worrying thread safety.
//
// We don't need to worry about duplicated module names here since the standard
// guarantees the module names should be unique to a program.
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

  bool isModuleUnitBuilt(llvm::StringRef ModuleName) const {
    return BuiltModuleNames.contains(ModuleName);
  }

  void addModuleFile(llvm::StringRef ModuleName,
                     llvm::StringRef ModuleFilePath) {
    RequiredModules.emplace_back(ModuleName, ModuleFilePath);
    BuiltModuleNames.insert(ModuleName);
  }

private:
  struct ModuleFile {
    ModuleFile(llvm::StringRef ModuleName, PathRef ModuleFilePath)
        : ModuleName(ModuleName.str()), ModuleFilePath(ModuleFilePath.str()) {}

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
  // A helper class to speedup the query if a module is built.
  llvm::StringSet<> BuiltModuleNames;
};

// Build a module file for module with `ModuleName`. The information of built
// module file are stored in \param BuiltModuleFiles.
llvm::Error buildModuleFile(llvm::StringRef ModuleName,
                            const GlobalCompilationDatabase &CDB,
                            const ThreadsafeFS &TFS, ProjectModules &MDB,
                            PathRef ModuleFilesPrefix,
                            StandalonePrerequisiteModules &BuiltModuleFiles) {
  if (BuiltModuleFiles.isModuleUnitBuilt(ModuleName))
    return llvm::Error::success();

  PathRef ModuleUnitFileName = MDB.getSourceForModuleName(ModuleName);
  // It is possible that we're meeting third party modules (modules whose
  // source are not in the project. e.g, the std module may be a third-party
  // module for most projects) or something wrong with the implementation of
  // ProjectModules.
  // FIXME: How should we treat third party modules here? If we want to ignore
  // third party modules, we should return true instead of false here.
  // Currently we simply bail out.
  if (ModuleUnitFileName.empty())
    return llvm::createStringError("Failed to get the primary source");

  // Try cheap operation earlier to boil-out cheaply if there are problems.
  auto Cmd = CDB.getCompileCommand(ModuleUnitFileName);
  if (!Cmd)
    return llvm::createStringError(
        llvm::formatv("No compile command for {0}", ModuleUnitFileName));

  for (auto &RequiredModuleName : MDB.getRequiredModules(ModuleUnitFileName)) {
    // Return early if there are errors building the module file.
    if (llvm::Error Err = buildModuleFile(RequiredModuleName, CDB, TFS, MDB,
                                          ModuleFilesPrefix, BuiltModuleFiles))
      return llvm::createStringError(
          llvm::formatv("Failed to build dependency {0}: {1}",
                        RequiredModuleName, llvm::toString(std::move(Err))));
  }

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

  BuiltModuleFiles.addModuleFile(ModuleName, Inputs.CompileCommand.Output);
  return llvm::Error::success();
}
} // namespace

std::unique_ptr<PrerequisiteModules>
ModulesBuilder::buildPrerequisiteModulesFor(PathRef File,
                                            const ThreadsafeFS &TFS) const {
  std::unique_ptr<ProjectModules> MDB = CDB.getProjectModules(File);
  if (!MDB) {
    elog("Failed to get Project Modules information for {0}", File);
    return std::make_unique<FailedPrerequisiteModules>();
  }

  std::vector<std::string> RequiredModuleNames = MDB->getRequiredModules(File);
  if (RequiredModuleNames.empty())
    return std::make_unique<StandalonePrerequisiteModules>();

  llvm::SmallString<256> ModuleFilesPrefix = getUniqueModuleFilesPath(File);

  log("Trying to build required modules for {0} in {1}", File,
      ModuleFilesPrefix);

  auto RequiredModules = std::make_unique<StandalonePrerequisiteModules>();

  for (llvm::StringRef RequiredModuleName : RequiredModuleNames) {
    // Return early if there is any error.
    if (llvm::Error Err =
            buildModuleFile(RequiredModuleName, CDB, TFS, *MDB.get(),
                            ModuleFilesPrefix, *RequiredModules.get())) {
      elog("Failed to build module {0}; due to {1}", RequiredModuleName,
           toString(std::move(Err)));
      return std::make_unique<FailedPrerequisiteModules>();
    }
  }

  log("Built required modules for {0} in {1}", File, ModuleFilesPrefix);

  return std::move(RequiredModules);
}

bool StandalonePrerequisiteModules::canReuse(
    const CompilerInvocation &CI,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS) const {
  if (RequiredModules.empty())
    return true;

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

  // Following the practice of clang's driver to suppres the checking for ODR
  // violation in GMF.
  // See
  // https://clang.llvm.org/docs/StandardCPlusPlusModules.html#object-definition-consistency
  // for example.
  Clang.getLangOpts().SkipODRCheckInGMF = true;

  Clang.createASTReader();
  for (auto &RequiredModule : RequiredModules) {
    llvm::StringRef BMIPath = RequiredModule.ModuleFilePath;
    // FIXME: Loading BMI fully is too heavy considering something cheaply to
    // check if we can reuse the BMI.
    auto ReadResult =
        Clang.getASTReader()->ReadAST(BMIPath, serialization::MK_MainFile,
                                      SourceLocation(), ASTReader::ARR_None);

    if (ReadResult != ASTReader::Success) {
      elog("Can't reuse {0}: {1}", BMIPath, ReadResult);
      return false;
    }
  }

  return true;
}

} // namespace clangd
} // namespace clang
