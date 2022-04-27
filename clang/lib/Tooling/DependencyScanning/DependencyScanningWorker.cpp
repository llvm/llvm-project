//===- DependencyScanningWorker.cpp - clang-scan-deps worker --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "clang/Tooling/Tooling.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

namespace {

/// Forwards the gatherered dependencies to the consumer.
class DependencyConsumerForwarder : public DependencyFileGenerator {
public:
  DependencyConsumerForwarder(std::unique_ptr<DependencyOutputOptions> Opts,
                              DependencyConsumer &C)
      : DependencyFileGenerator(*Opts), Opts(std::move(Opts)), C(C) {}

  void finishedMainFile(DiagnosticsEngine &Diags) override {
    C.handleDependencyOutputOpts(*Opts);
    llvm::SmallString<256> CanonPath;
    for (const auto &File : getDependencies()) {
      CanonPath = File;
      llvm::sys::path::remove_dots(CanonPath, /*remove_dot_dot=*/true);
      C.handleFileDependency(CanonPath);
    }
  }

private:
  std::unique_ptr<DependencyOutputOptions> Opts;
  DependencyConsumer &C;
};

using PrebuiltModuleFilesT = decltype(HeaderSearchOptions::PrebuiltModuleFiles);

/// A listener that collects the imported modules and optionally the input
/// files.
class PrebuiltModuleListener : public ASTReaderListener {
public:
  PrebuiltModuleListener(PrebuiltModuleFilesT &PrebuiltModuleFiles,
                         llvm::StringSet<> &InputFiles, bool VisitInputFiles,
                         llvm::SmallVector<std::string> &NewModuleFiles)
      : PrebuiltModuleFiles(PrebuiltModuleFiles), InputFiles(InputFiles),
        VisitInputFiles(VisitInputFiles), NewModuleFiles(NewModuleFiles) {}

  bool needsImportVisitation() const override { return true; }
  bool needsInputFileVisitation() override { return VisitInputFiles; }
  bool needsSystemInputFileVisitation() override { return VisitInputFiles; }

  void visitImport(StringRef ModuleName, StringRef Filename) override {
    if (PrebuiltModuleFiles.insert({ModuleName.str(), Filename.str()}).second)
      NewModuleFiles.push_back(Filename.str());
  }

  bool visitInputFile(StringRef Filename, bool isSystem, bool isOverridden,
                      bool isExplicitModule) override {
    InputFiles.insert(Filename);
    return true;
  }

private:
  PrebuiltModuleFilesT &PrebuiltModuleFiles;
  llvm::StringSet<> &InputFiles;
  bool VisitInputFiles;
  llvm::SmallVector<std::string> &NewModuleFiles;
};

/// Visit the given prebuilt module and collect all of the modules it
/// transitively imports and contributing input files.
static void visitPrebuiltModule(StringRef PrebuiltModuleFilename,
                                CompilerInstance &CI,
                                PrebuiltModuleFilesT &ModuleFiles,
                                llvm::StringSet<> &InputFiles,
                                bool VisitInputFiles) {
  // List of module files to be processed.
  llvm::SmallVector<std::string> Worklist{PrebuiltModuleFilename.str()};
  PrebuiltModuleListener Listener(ModuleFiles, InputFiles, VisitInputFiles,
                                  Worklist);

  while (!Worklist.empty())
    ASTReader::readASTFileControlBlock(
        Worklist.pop_back_val(), CI.getFileManager(),
        CI.getPCHContainerReader(),
        /*FindModuleFileExtensions=*/false, Listener,
        /*ValidateDiagnosticOptions=*/false);
}

/// Transform arbitrary file name into an object-like file name.
static std::string makeObjFileName(StringRef FileName) {
  SmallString<128> ObjFileName(FileName);
  llvm::sys::path::replace_extension(ObjFileName, "o");
  return std::string(ObjFileName.str());
}

/// Deduce the dependency target based on the output file and input files.
static std::string
deduceDepTarget(const std::string &OutputFile,
                const SmallVectorImpl<FrontendInputFile> &InputFiles) {
  if (OutputFile != "-")
    return OutputFile;

  if (InputFiles.empty() || !InputFiles.front().isFile())
    return "clang-scan-deps\\ dependency";

  return makeObjFileName(InputFiles.front().getFile());
}

/// Sanitize diagnostic options for dependency scan.
static void sanitizeDiagOpts(DiagnosticOptions &DiagOpts) {
  // Don't print 'X warnings and Y errors generated'.
  DiagOpts.ShowCarets = false;
  // Don't write out diagnostic file.
  DiagOpts.DiagnosticSerializationFile.clear();
  // Don't treat warnings as errors.
  DiagOpts.Warnings.push_back("no-error");
}

/// A clang tool that runs the preprocessor in a mode that's optimized for
/// dependency scanning for the given compiler invocation.
class DependencyScanningAction : public tooling::ToolAction {
public:
  DependencyScanningAction(
      StringRef WorkingDirectory, DependencyConsumer &Consumer,
      llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS,
      ExcludedPreprocessorDirectiveSkipMapping &PPSkipMappings,
      ScanningOutputFormat Format, bool OptimizeArgs,
      llvm::Optional<StringRef> ModuleName = None)
      : WorkingDirectory(WorkingDirectory), Consumer(Consumer),
        DepFS(std::move(DepFS)), PPSkipMappings(PPSkipMappings), Format(Format),
        OptimizeArgs(OptimizeArgs), ModuleName(ModuleName) {}

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *FileMgr,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    // Make a deep copy of the original Clang invocation.
    CompilerInvocation OriginalInvocation(*Invocation);

    // Create a compiler instance to handle the actual work.
    CompilerInstance ScanInstance(std::move(PCHContainerOps));
    ScanInstance.setInvocation(std::move(Invocation));

    // Create the compiler's actual diagnostics engine.
    sanitizeDiagOpts(ScanInstance.getDiagnosticOpts());
    ScanInstance.createDiagnostics(DiagConsumer, /*ShouldOwnClient=*/false);
    if (!ScanInstance.hasDiagnostics())
      return false;

    ScanInstance.getPreprocessorOpts().AllowPCHWithDifferentModulesCachePath =
        true;

    ScanInstance.getFrontendOpts().GenerateGlobalModuleIndex = false;
    ScanInstance.getFrontendOpts().UseGlobalModuleIndex = false;

    FileMgr->getFileSystemOpts().WorkingDir = std::string(WorkingDirectory);
    ScanInstance.setFileManager(FileMgr);
    ScanInstance.createSourceManager(*FileMgr);

    llvm::StringSet<> PrebuiltModulesInputFiles;
    // Store the list of prebuilt module files into header search options. This
    // will prevent the implicit build to create duplicate modules and will
    // force reuse of the existing prebuilt module files instead.
    if (!ScanInstance.getPreprocessorOpts().ImplicitPCHInclude.empty())
      visitPrebuiltModule(
          ScanInstance.getPreprocessorOpts().ImplicitPCHInclude, ScanInstance,
          ScanInstance.getHeaderSearchOpts().PrebuiltModuleFiles,
          PrebuiltModulesInputFiles, /*VisitInputFiles=*/DepFS != nullptr);

    // Use the dependency scanning optimized file system if requested to do so.
    if (DepFS) {
      DepFS->enableMinimizationOfAllFiles();
      // Don't minimize any files that contributed to prebuilt modules. The
      // implicit build validates the modules by comparing the reported sizes of
      // their inputs to the current state of the filesystem. Minimization would
      // throw this mechanism off.
      for (const auto &File : PrebuiltModulesInputFiles)
        DepFS->disableMinimization(File.getKey());
      // Don't minimize any files that were explicitly passed in the build
      // settings and that might be opened.
      for (const auto &E : ScanInstance.getHeaderSearchOpts().UserEntries)
        DepFS->disableMinimization(E.Path);
      for (const auto &F : ScanInstance.getHeaderSearchOpts().VFSOverlayFiles)
        DepFS->disableMinimization(F);

      // Support for virtual file system overlays on top of the caching
      // filesystem.
      FileMgr->setVirtualFileSystem(createVFSFromCompilerInvocation(
          ScanInstance.getInvocation(), ScanInstance.getDiagnostics(), DepFS));

      // Pass the skip mappings which should speed up excluded conditional block
      // skipping in the preprocessor.
      ScanInstance.getPreprocessorOpts()
          .ExcludedConditionalDirectiveSkipMappings = &PPSkipMappings;
    }

    // Create the dependency collector that will collect the produced
    // dependencies.
    //
    // This also moves the existing dependency output options from the
    // invocation to the collector. The options in the invocation are reset,
    // which ensures that the compiler won't create new dependency collectors,
    // and thus won't write out the extra '.d' files to disk.
    auto Opts = std::make_unique<DependencyOutputOptions>();
    std::swap(*Opts, ScanInstance.getInvocation().getDependencyOutputOpts());
    // We need at least one -MT equivalent for the generator of make dependency
    // files to work.
    if (Opts->Targets.empty())
      Opts->Targets = {
          deduceDepTarget(ScanInstance.getFrontendOpts().OutputFile,
                          ScanInstance.getFrontendOpts().Inputs)};
    Opts->IncludeSystemHeaders = true;

    switch (Format) {
    case ScanningOutputFormat::Make:
      ScanInstance.addDependencyCollector(
          std::make_shared<DependencyConsumerForwarder>(std::move(Opts),
                                                        Consumer));
      break;
    case ScanningOutputFormat::Full:
      ScanInstance.addDependencyCollector(std::make_shared<ModuleDepCollector>(
          std::move(Opts), ScanInstance, Consumer,
          std::move(OriginalInvocation), OptimizeArgs));
      break;
    }

    // Consider different header search and diagnostic options to create
    // different modules. This avoids the unsound aliasing of module PCMs.
    //
    // TODO: Implement diagnostic bucketing to reduce the impact of strict
    // context hashing.
    ScanInstance.getHeaderSearchOpts().ModulesStrictContextHash = true;

    std::unique_ptr<FrontendAction> Action;

    if (ModuleName.hasValue())
      Action = std::make_unique<GetDependenciesByModuleNameAction>(*ModuleName);
    else
      Action = std::make_unique<ReadPCHAndPreprocessAction>();

    const bool Result = ScanInstance.ExecuteAction(*Action);
    if (!DepFS)
      FileMgr->clearStatCache();
    return Result;
  }

private:
  StringRef WorkingDirectory;
  DependencyConsumer &Consumer;
  llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;
  ExcludedPreprocessorDirectiveSkipMapping &PPSkipMappings;
  ScanningOutputFormat Format;
  bool OptimizeArgs;
  llvm::Optional<StringRef> ModuleName;
};

} // end anonymous namespace

DependencyScanningWorker::DependencyScanningWorker(
    DependencyScanningService &Service)
    : Format(Service.getFormat()), OptimizeArgs(Service.canOptimizeArgs()) {
  PCHContainerOps = std::make_shared<PCHContainerOperations>();
  PCHContainerOps->registerReader(
      std::make_unique<ObjectFilePCHContainerReader>());
  // We don't need to write object files, but the current PCH implementation
  // requires the writer to be registered as well.
  PCHContainerOps->registerWriter(
      std::make_unique<ObjectFilePCHContainerWriter>());

  auto OverlayFS = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(
      llvm::vfs::createPhysicalFileSystem());
  InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  OverlayFS->pushOverlay(InMemoryFS);
  RealFS = OverlayFS;

  if (Service.getMode() == ScanningMode::MinimizedSourcePreprocessing)
    DepFS = new DependencyScanningWorkerFilesystem(Service.getSharedCache(),
                                                   RealFS, PPSkipMappings);
  if (Service.canReuseFileManager())
    Files = new FileManager(FileSystemOptions(), RealFS);
}

static llvm::Error
runWithDiags(DiagnosticOptions *DiagOpts,
             llvm::function_ref<bool(DiagnosticConsumer &, DiagnosticOptions &)>
                 BodyShouldSucceed) {
  sanitizeDiagOpts(*DiagOpts);

  // Capture the emitted diagnostics and report them to the client
  // in the case of a failure.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  TextDiagnosticPrinter DiagPrinter(DiagnosticsOS, DiagOpts);

  if (BodyShouldSucceed(DiagPrinter, *DiagOpts))
    return llvm::Error::success();
  return llvm::make_error<llvm::StringError>(DiagnosticsOS.str(),
                                             llvm::inconvertibleErrorCode());
}

llvm::Error DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, llvm::Optional<StringRef> ModuleName) {
  // Reset what might have been modified in the previous worker invocation.
  RealFS->setCurrentWorkingDirectory(WorkingDirectory);
  if (Files)
    Files->setVirtualFileSystem(RealFS);

  llvm::IntrusiveRefCntPtr<FileManager> CurrentFiles =
      Files ? Files : new FileManager(FileSystemOptions(), RealFS);

  Optional<std::vector<std::string>> ModifiedCommandLine;
  if (ModuleName.hasValue()) {
    ModifiedCommandLine = CommandLine;
    InMemoryFS->addFile(*ModuleName, 0, llvm::MemoryBuffer::getMemBuffer(""));
    ModifiedCommandLine->emplace_back(*ModuleName);
  }

  const std::vector<std::string> &FinalCommandLine =
      ModifiedCommandLine ? *ModifiedCommandLine : CommandLine;

  std::vector<const char *> FinalCCommandLine(CommandLine.size(), nullptr);
  llvm::transform(CommandLine, FinalCCommandLine.begin(),
                  [](const std::string &Str) { return Str.c_str(); });

  return runWithDiags(CreateAndPopulateDiagOpts(FinalCCommandLine).release(),
                      [&](DiagnosticConsumer &DC, DiagnosticOptions &DiagOpts) {
                        DependencyScanningAction Action(
                            WorkingDirectory, Consumer, DepFS, PPSkipMappings,
                            Format, OptimizeArgs, ModuleName);
                        // Create an invocation that uses the underlying file
                        // system to ensure that any file system requests that
                        // are made by the driver do not go through the
                        // dependency scanning filesystem.
                        ToolInvocation Invocation(FinalCommandLine, &Action,
                                                  CurrentFiles.get(),
                                                  PCHContainerOps);
                        Invocation.setDiagnosticConsumer(&DC);
                        Invocation.setDiagnosticOptions(&DiagOpts);
                        return Invocation.run();
                      });
}
