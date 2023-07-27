//===- DependencyScanningWorker.cpp - clang-scan-deps worker --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Host.h"
#include <optional>

using namespace clang;
using namespace tooling;
using namespace dependencies;

namespace {

/// Forwards the gatherered dependencies to the consumer.
class DependencyConsumerForwarder : public DependencyFileGenerator {
public:
  DependencyConsumerForwarder(std::unique_ptr<DependencyOutputOptions> Opts,
                              StringRef WorkingDirectory, DependencyConsumer &C)
      : DependencyFileGenerator(*Opts), WorkingDirectory(WorkingDirectory),
        Opts(std::move(Opts)), C(C) {}

  void finishedMainFile(DiagnosticsEngine &Diags) override {
    C.handleDependencyOutputOpts(*Opts);
    llvm::SmallString<256> CanonPath;
    for (const auto &File : getDependencies()) {
      CanonPath = File;
      llvm::sys::path::remove_dots(CanonPath, /*remove_dot_dot=*/true);
      llvm::sys::fs::make_absolute(WorkingDirectory, CanonPath);
      C.handleFileDependency(CanonPath);
    }
  }

private:
  StringRef WorkingDirectory;
  std::unique_ptr<DependencyOutputOptions> Opts;
  DependencyConsumer &C;
};

using PrebuiltModuleFilesT = decltype(HeaderSearchOptions::PrebuiltModuleFiles);

/// A listener that collects the imported modules and optionally the input
/// files.
class PrebuiltModuleListener : public ASTReaderListener {
public:
  PrebuiltModuleListener(PrebuiltModuleFilesT &PrebuiltModuleFiles,
                         llvm::SmallVector<std::string> &NewModuleFiles)
      : PrebuiltModuleFiles(PrebuiltModuleFiles),
        NewModuleFiles(NewModuleFiles) {}

  bool needsImportVisitation() const override { return true; }

  void visitImport(StringRef ModuleName, StringRef Filename) override {
    if (PrebuiltModuleFiles.insert({ModuleName.str(), Filename.str()}).second)
      NewModuleFiles.push_back(Filename.str());
  }

private:
  PrebuiltModuleFilesT &PrebuiltModuleFiles;
  llvm::SmallVector<std::string> &NewModuleFiles;
};

/// Visit the given prebuilt module and collect all of the modules it
/// transitively imports and contributing input files.
static void visitPrebuiltModule(StringRef PrebuiltModuleFilename,
                                CompilerInstance &CI,
                                PrebuiltModuleFilesT &ModuleFiles) {
  // List of module files to be processed.
  llvm::SmallVector<std::string> Worklist{PrebuiltModuleFilename.str()};
  PrebuiltModuleListener Listener(ModuleFiles, Worklist);

  while (!Worklist.empty())
    ASTReader::readASTFileControlBlock(
        Worklist.pop_back_val(), CI.getFileManager(), CI.getModuleCache(),
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
  // Don't emit warnings as errors (and all other warnings too).
  DiagOpts.IgnoreWarnings = true;
}

/// A clang tool that runs the preprocessor in a mode that's optimized for
/// dependency scanning for the given compiler invocation.
class DependencyScanningAction : public tooling::ToolAction {
public:
  DependencyScanningAction(
      StringRef WorkingDirectory, DependencyConsumer &Consumer,
      DependencyActionController &Controller,
      llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS,
      ScanningOutputFormat Format, bool OptimizeArgs, bool EagerLoadModules,
      bool DisableFree, std::optional<StringRef> ModuleName = std::nullopt)
      : WorkingDirectory(WorkingDirectory), Consumer(Consumer),
        Controller(Controller), DepFS(std::move(DepFS)), Format(Format),
        OptimizeArgs(OptimizeArgs), EagerLoadModules(EagerLoadModules),
        DisableFree(DisableFree), ModuleName(ModuleName) {}

  bool runInvocation(std::shared_ptr<CompilerInvocation> Invocation,
                     FileManager *FileMgr,
                     std::shared_ptr<PCHContainerOperations> PCHContainerOps,
                     DiagnosticConsumer *DiagConsumer) override {
    // Make a deep copy of the original Clang invocation.
    CompilerInvocation OriginalInvocation(*Invocation);
    // Restore the value of DisableFree, which may be modified by Tooling.
    OriginalInvocation.getFrontendOpts().DisableFree = DisableFree;

    if (Scanned) {
      // Scanning runs once for the first -cc1 invocation in a chain of driver
      // jobs. For any dependent jobs, reuse the scanning result and just
      // update the LastCC1Arguments to correspond to the new invocation.
      // FIXME: to support multi-arch builds, each arch requires a separate scan
      setLastCC1Arguments(std::move(OriginalInvocation));
      return true;
    }

    Scanned = true;

    // Create a compiler instance to handle the actual work.
    ScanInstanceStorage.emplace(std::move(PCHContainerOps));
    CompilerInstance &ScanInstance = *ScanInstanceStorage;
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
    ScanInstance.getFrontendOpts().ModulesShareFileManager = false;
    ScanInstance.getHeaderSearchOpts().ModuleFormat = "raw";

    ScanInstance.setFileManager(FileMgr);
    // Support for virtual file system overlays.
    FileMgr->setVirtualFileSystem(createVFSFromCompilerInvocation(
        ScanInstance.getInvocation(), ScanInstance.getDiagnostics(),
        FileMgr->getVirtualFileSystemPtr()));

    ScanInstance.createSourceManager(*FileMgr);

    // Store the list of prebuilt module files into header search options. This
    // will prevent the implicit build to create duplicate modules and will
    // force reuse of the existing prebuilt module files instead.
    if (!ScanInstance.getPreprocessorOpts().ImplicitPCHInclude.empty())
      visitPrebuiltModule(
          ScanInstance.getPreprocessorOpts().ImplicitPCHInclude, ScanInstance,
          ScanInstance.getHeaderSearchOpts().PrebuiltModuleFiles);

    // Use the dependency scanning optimized file system if requested to do so.
    if (DepFS) {
      llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> LocalDepFS =
          DepFS;
      ScanInstance.getPreprocessorOpts().DependencyDirectivesForFile =
          [LocalDepFS = std::move(LocalDepFS)](FileEntryRef File)
          -> std::optional<ArrayRef<dependency_directives_scan::Directive>> {
        if (llvm::ErrorOr<EntryRef> Entry =
                LocalDepFS->getOrCreateFileSystemEntry(File.getName()))
          return Entry->getDirectiveTokens();
        return std::nullopt;
      };
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
          std::make_shared<DependencyConsumerForwarder>(
              std::move(Opts), WorkingDirectory, Consumer));
      break;
    case ScanningOutputFormat::P1689:
    case ScanningOutputFormat::Full:
      MDC = std::make_shared<ModuleDepCollector>(
          std::move(Opts), ScanInstance, Consumer, Controller,
          OriginalInvocation, OptimizeArgs, EagerLoadModules,
          Format == ScanningOutputFormat::P1689);
      ScanInstance.addDependencyCollector(MDC);
      break;
    }

    // Consider different header search and diagnostic options to create
    // different modules. This avoids the unsound aliasing of module PCMs.
    //
    // TODO: Implement diagnostic bucketing to reduce the impact of strict
    // context hashing.
    ScanInstance.getHeaderSearchOpts().ModulesStrictContextHash = true;

    // Avoid some checks and module map parsing when loading PCM files.
    ScanInstance.getPreprocessorOpts().ModulesCheckRelocated = false;

    std::unique_ptr<FrontendAction> Action;

    if (ModuleName)
      Action = std::make_unique<GetDependenciesByModuleNameAction>(*ModuleName);
    else
      Action = std::make_unique<ReadPCHAndPreprocessAction>();

    const bool Result = ScanInstance.ExecuteAction(*Action);

    if (Result)
      setLastCC1Arguments(std::move(OriginalInvocation));

    return Result;
  }

  bool hasScanned() const { return Scanned; }

  /// Take the cc1 arguments corresponding to the most recent invocation used
  /// with this action. Any modifications implied by the discovered dependencies
  /// will have already been applied.
  std::vector<std::string> takeLastCC1Arguments() {
    std::vector<std::string> Result;
    std::swap(Result, LastCC1Arguments); // Reset LastCC1Arguments to empty.
    return Result;
  }

private:
  void setLastCC1Arguments(CompilerInvocation &&CI) {
    if (MDC)
      MDC->applyDiscoveredDependencies(CI);
    LastCC1Arguments = CI.getCC1CommandLine();
  }

private:
  StringRef WorkingDirectory;
  DependencyConsumer &Consumer;
  DependencyActionController &Controller;
  llvm::IntrusiveRefCntPtr<DependencyScanningWorkerFilesystem> DepFS;
  ScanningOutputFormat Format;
  bool OptimizeArgs;
  bool EagerLoadModules;
  bool DisableFree;
  std::optional<StringRef> ModuleName;
  std::optional<CompilerInstance> ScanInstanceStorage;
  std::shared_ptr<ModuleDepCollector> MDC;
  std::vector<std::string> LastCC1Arguments;
  bool Scanned = false;
};

} // end anonymous namespace

DependencyScanningWorker::DependencyScanningWorker(
    DependencyScanningService &Service,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : Format(Service.getFormat()), OptimizeArgs(Service.canOptimizeArgs()),
      EagerLoadModules(Service.shouldEagerLoadModules()) {
  PCHContainerOps = std::make_shared<PCHContainerOperations>();
  // We need to read object files from PCH built outside the scanner.
  PCHContainerOps->registerReader(
      std::make_unique<ObjectFilePCHContainerReader>());
  // The scanner itself writes only raw ast files.
  PCHContainerOps->registerWriter(std::make_unique<RawPCHContainerWriter>());

  switch (Service.getMode()) {
  case ScanningMode::DependencyDirectivesScan:
    DepFS =
        new DependencyScanningWorkerFilesystem(Service.getSharedCache(), FS);
    BaseFS = DepFS;
    break;
  case ScanningMode::CanonicalPreprocessing:
    DepFS = nullptr;
    BaseFS = FS;
    break;
  }
}

llvm::Error DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    std::optional<StringRef> ModuleName) {
  std::vector<const char *> CLI;
  for (const std::string &Arg : CommandLine)
    CLI.push_back(Arg.c_str());
  auto DiagOpts = CreateAndPopulateDiagOpts(CLI);
  sanitizeDiagOpts(*DiagOpts);

  // Capture the emitted diagnostics and report them to the client
  // in the case of a failure.
  std::string DiagnosticOutput;
  llvm::raw_string_ostream DiagnosticsOS(DiagnosticOutput);
  TextDiagnosticPrinter DiagPrinter(DiagnosticsOS, DiagOpts.release());

  if (computeDependencies(WorkingDirectory, CommandLine, Consumer, Controller,
                          DiagPrinter, ModuleName))
    return llvm::Error::success();
  return llvm::make_error<llvm::StringError>(DiagnosticsOS.str(),
                                             llvm::inconvertibleErrorCode());
}

static bool forEachDriverJob(
    ArrayRef<std::string> ArgStrs, DiagnosticsEngine &Diags, FileManager &FM,
    llvm::function_ref<bool(const driver::Command &Cmd)> Callback) {
  SmallVector<const char *, 256> Argv;
  Argv.reserve(ArgStrs.size());
  for (const std::string &Arg : ArgStrs)
    Argv.push_back(Arg.c_str());

  llvm::vfs::FileSystem *FS = &FM.getVirtualFileSystem();

  std::unique_ptr<driver::Driver> Driver = std::make_unique<driver::Driver>(
      Argv[0], llvm::sys::getDefaultTargetTriple(), Diags,
      "clang LLVM compiler", FS);
  Driver->setTitle("clang_based_tool");

  llvm::BumpPtrAllocator Alloc;
  bool CLMode = driver::IsClangCL(
      driver::getDriverMode(Argv[0], ArrayRef(Argv).slice(1)));

  if (llvm::Error E = driver::expandResponseFiles(Argv, CLMode, Alloc, FS)) {
    Diags.Report(diag::err_drv_expand_response_file)
        << llvm::toString(std::move(E));
    return false;
  }

  const std::unique_ptr<driver::Compilation> Compilation(
      Driver->BuildCompilation(llvm::ArrayRef(Argv)));
  if (!Compilation)
    return false;

  for (const driver::Command &Job : Compilation->getJobs()) {
    if (!Callback(Job))
      return false;
  }
  return true;
}

bool DependencyScanningWorker::computeDependencies(
    StringRef WorkingDirectory, const std::vector<std::string> &CommandLine,
    DependencyConsumer &Consumer, DependencyActionController &Controller,
    DiagnosticConsumer &DC, std::optional<StringRef> ModuleName) {
  // Reset what might have been modified in the previous worker invocation.
  BaseFS->setCurrentWorkingDirectory(WorkingDirectory);

  std::optional<std::vector<std::string>> ModifiedCommandLine;
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> ModifiedFS;

  // If we're scanning based on a module name alone, we don't expect the client
  // to provide us with an input file. However, the driver really wants to have
  // one. Let's just make it up to make the driver happy.
  if (ModuleName) {
    auto OverlayFS =
        llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(BaseFS);
    auto InMemoryFS =
        llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    InMemoryFS->setCurrentWorkingDirectory(WorkingDirectory);
    OverlayFS->pushOverlay(InMemoryFS);
    ModifiedFS = OverlayFS;

    SmallString<128> FakeInputPath;
    // TODO: We should retry the creation if the path already exists.
    llvm::sys::fs::createUniquePath(*ModuleName + "-%%%%%%%%.input",
                                    FakeInputPath,
                                    /*MakeAbsolute=*/false);
    InMemoryFS->addFile(FakeInputPath, 0, llvm::MemoryBuffer::getMemBuffer(""));

    ModifiedCommandLine = CommandLine;
    ModifiedCommandLine->emplace_back(FakeInputPath);
  }

  const std::vector<std::string> &FinalCommandLine =
      ModifiedCommandLine ? *ModifiedCommandLine : CommandLine;
  auto &FinalFS = ModifiedFS ? ModifiedFS : BaseFS;

  FileSystemOptions FSOpts;
  FSOpts.WorkingDir = WorkingDirectory.str();
  auto FileMgr = llvm::makeIntrusiveRefCnt<FileManager>(FSOpts, FinalFS);

  std::vector<const char *> FinalCCommandLine(FinalCommandLine.size(), nullptr);
  llvm::transform(FinalCommandLine, FinalCCommandLine.begin(),
                  [](const std::string &Str) { return Str.c_str(); });

  auto DiagOpts = CreateAndPopulateDiagOpts(FinalCCommandLine);
  sanitizeDiagOpts(*DiagOpts);
  IntrusiveRefCntPtr<DiagnosticsEngine> Diags =
      CompilerInstance::createDiagnostics(DiagOpts.release(), &DC,
                                          /*ShouldOwnClient=*/false);

  // Although `Diagnostics` are used only for command-line parsing, the
  // custom `DiagConsumer` might expect a `SourceManager` to be present.
  SourceManager SrcMgr(*Diags, *FileMgr);
  Diags->setSourceManager(&SrcMgr);
  // DisableFree is modified by Tooling for running
  // in-process; preserve the original value, which is
  // always true for a driver invocation.
  bool DisableFree = true;
  DependencyScanningAction Action(WorkingDirectory, Consumer, Controller, DepFS,
                                  Format, OptimizeArgs, EagerLoadModules,
                                  DisableFree, ModuleName);
  bool Success = forEachDriverJob(
      FinalCommandLine, *Diags, *FileMgr, [&](const driver::Command &Cmd) {
        if (StringRef(Cmd.getCreator().getName()) != "clang") {
          // Non-clang command. Just pass through to the dependency
          // consumer.
          Consumer.handleBuildCommand(
              {Cmd.getExecutable(),
               {Cmd.getArguments().begin(), Cmd.getArguments().end()}});
          return true;
        }

        std::vector<std::string> Argv;
        Argv.push_back(Cmd.getExecutable());
        Argv.insert(Argv.end(), Cmd.getArguments().begin(),
                    Cmd.getArguments().end());

        // Create an invocation that uses the underlying file
        // system to ensure that any file system requests that
        // are made by the driver do not go through the
        // dependency scanning filesystem.
        ToolInvocation Invocation(std::move(Argv), &Action, &*FileMgr,
                                  PCHContainerOps);
        Invocation.setDiagnosticConsumer(Diags->getClient());
        Invocation.setDiagnosticOptions(&Diags->getDiagnosticOptions());
        if (!Invocation.run())
          return false;

        std::vector<std::string> Args = Action.takeLastCC1Arguments();
        Consumer.handleBuildCommand({Cmd.getExecutable(), std::move(Args)});
        return true;
      });

  if (Success && !Action.hasScanned())
    Diags->Report(diag::err_fe_expected_compiler_job)
        << llvm::join(FinalCommandLine, " ");
  return Success && Action.hasScanned();
}

DependencyActionController::~DependencyActionController() {}
