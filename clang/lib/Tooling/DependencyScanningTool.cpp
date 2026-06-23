//===- DependencyScanningTool.cpp - clang-scan-deps service ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanningTool.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/CAS/IncludeTree.h"
#include "clang/DependencyScanning/CachingActions.h"
#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/iterator.h"
#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/TargetParser/Host.h"
#include <optional>

using namespace clang;
using namespace tooling;
using namespace dependencies;
using llvm::Error;

namespace {
/// Prints out all of the gathered dependencies into a string.
class MakeDependencyPrinterConsumer : public DependencyConsumer {
public:
  void handleBuildCommand(Command) override {}

  void
  handleDependencyOutputOpts(const DependencyOutputOptions &Opts) override {
    this->Opts = std::make_unique<DependencyOutputOptions>(Opts);
  }

  void handleFileDependency(StringRef File) override {
    SmallString<128> NormalizedFile = File;
    llvm::sys::path::remove_dots(NormalizedFile, /*remove_dot_dot=*/true);
    Dependencies.emplace_back(NormalizedFile.str());
  }

  // These are ignored for the make format as it can't support the full
  // set of deps, and handleFileDependency handles enough for implicitly
  // built modules to work.
  void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) override {}
  void handleModuleDependency(ModuleDeps MD) override {
    MD.forEachFileDep([this](StringRef File) {
      DependenciesFromModules.push_back(std::string(File));
    });
  }
  void handleDirectModuleDependency(ModuleID ID) override {}
  void handleVisibleModule(std::string ModuleName) override {}
  void handleContextHash(std::string Hash) override {}

  void printDependencies(std::string &S) {
    assert(Opts && "Handled dependency output options.");

    class DependencyPrinter : public DependencyFileGenerator {
    public:
      DependencyPrinter(DependencyOutputOptions &Opts,
                        ArrayRef<std::string> Dependencies,
                        ArrayRef<std::string> ModuleDependencies)
          : DependencyFileGenerator(Opts) {
        for (const auto &Dep : Dependencies)
          addDependency(Dep);
        for (const auto &Dep : ModuleDependencies)
          addDependency(Dep);
      }

      void printDependencies(std::string &S) {
        llvm::raw_string_ostream OS(S);
        outputDependencyFile(OS);
      }
    };

    DependencyPrinter Generator(*Opts, Dependencies, DependenciesFromModules);
    Generator.printDependencies(S);
  }

protected:
  std::unique_ptr<DependencyOutputOptions> Opts;
  std::vector<std::string> Dependencies;
  std::vector<std::string> DependenciesFromModules;
};
} // anonymous namespace

static std::pair<std::unique_ptr<driver::Driver>,
                 std::unique_ptr<driver::Compilation>>
buildCompilation(ArrayRef<std::string> ArgStrs, DiagnosticsEngine &Diags,
                 IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                 llvm::BumpPtrAllocator &Alloc) {
  SmallVector<const char *, 256> Argv;
  Argv.reserve(ArgStrs.size());
  for (const std::string &Arg : ArgStrs)
    Argv.push_back(Arg.c_str());

  std::unique_ptr<driver::Driver> Driver = std::make_unique<driver::Driver>(
      Argv[0], llvm::sys::getDefaultTargetTriple(), Diags,
      "clang LLVM compiler", FS);
  Driver->setTitle("clang_based_tool");

  bool CLMode = driver::IsClangCL(
      driver::getDriverMode(Argv[0], ArrayRef(Argv).slice(1)));

  if (llvm::Error E =
          driver::expandResponseFiles(Argv, CLMode, Alloc, FS.get())) {
    Diags.Report(diag::err_drv_expand_response_file)
        << llvm::toString(std::move(E));
    return std::make_pair(nullptr, nullptr);
  }

  std::unique_ptr<driver::Compilation> Compilation(
      Driver->BuildCompilation(Argv));
  if (!Compilation)
    return std::make_pair(nullptr, nullptr);

  if (Compilation->containsError())
    return std::make_pair(nullptr, nullptr);

  if (Compilation->getJobs().empty()) {
    Diags.Report(diag::err_fe_expected_compiler_job)
        << llvm::join(ArgStrs, " ");
    return std::make_pair(nullptr, nullptr);
  }

  return std::make_pair(std::move(Driver), std::move(Compilation));
}

/// Constructs the full frontend command line, including executable, for the
/// given driver \c Cmd.
static SmallVector<std::string, 0>
buildCC1CommandLine(const driver::Command &Cmd) {
  const auto &Args = Cmd.getArguments();
  SmallVector<std::string, 0> Out;
  Out.reserve(Args.size() + 1);
  Out.emplace_back(Cmd.getExecutable());
  llvm::append_range(Out, Args);
  return Out;
}

static bool computeDependenciesForDriverCommandLine(
    DependencyScanningWorker &Worker, StringRef WorkingDirectory,
    ArrayRef<std::string> CommandLine, DependencyConsumer &Consumer,
    DependencyActionController &Controller, DiagnosticConsumer &DiagConsumer,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS) {
  auto FS = Worker.makeEffectiveVFS(WorkingDirectory, OverlayFS);

  // Compilation holds a non-owning a reference to the Driver, hence we need to
  // keep the Driver alive when we use Compilation. Arguments to commands may be
  // owned by Alloc when expanded from response files.
  llvm::BumpPtrAllocator Alloc;
  auto DiagEngineWithDiagOpts =
      DiagnosticsEngineWithDiagOpts(CommandLine, FS, DiagConsumer);
  const auto [Driver, Compilation] = buildCompilation(
      CommandLine, *DiagEngineWithDiagOpts.DiagEngine, FS, Alloc);
  if (!Compilation)
    return false;

  SmallVector<SmallVector<std::string, 0>> FrontendCommandLines;
  for (const auto &Cmd : Compilation->getJobs())
    FrontendCommandLines.push_back(buildCC1CommandLine(Cmd));
  SmallVector<ArrayRef<std::string>> FrontendCommandLinesView(
      FrontendCommandLines.begin(), FrontendCommandLines.end());

  return Worker.computeDependencies(WorkingDirectory, FrontendCommandLinesView,
                                    Consumer, Controller, DiagConsumer,
                                    std::move(OverlayFS));
}

static llvm::Error makeErrorFromDiagnosticsOS(
    TextDiagnosticsPrinterWithOutput &DiagPrinterWithOS) {
  return llvm::make_error<llvm::StringError>(
      DiagPrinterWithOS.DiagnosticsOS.str(), llvm::inconvertibleErrorCode());
}

bool tooling::computeDependencies(
    DependencyScanningWorker &Worker, StringRef WorkingDirectory,
    ArrayRef<std::string> CommandLine, DependencyConsumer &Consumer,
    DependencyActionController &Controller, DiagnosticConsumer &DiagConsumer,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS) {
  const auto IsCC1Input = (CommandLine.size() >= 2 && CommandLine[1] == "-cc1");
  return IsCC1Input ? Worker.computeDependencies(WorkingDirectory, CommandLine,
                                                 Consumer, Controller,
                                                 DiagConsumer, OverlayFS)
                    : computeDependenciesForDriverCommandLine(
                          Worker, WorkingDirectory, CommandLine, Consumer,
                          Controller, DiagConsumer, OverlayFS);
}

std::optional<std::string> DependencyScanningTool::getDependencyFile(
    ArrayRef<std::string> CommandLine, StringRef CWD,
    LookupModuleOutputCallback LookupModuleOutput,
    DiagnosticConsumer &DiagConsumer) {
  MakeDependencyPrinterConsumer DepConsumer;
  CallbackActionController Controller(LookupModuleOutput);
  if (!computeDependencies(Worker, CWD, CommandLine, DepConsumer, Controller,
                           DiagConsumer))
    return std::nullopt;
  std::string Output;
  DepConsumer.printDependencies(Output);
  return Output;
}

namespace {
class EmptyDependencyConsumer : public DependencyConsumer {
  void
  handleDependencyOutputOpts(const DependencyOutputOptions &Opts) override {}

  void handleFileDependency(StringRef Filename) override {}

  void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) override {}

  void handleModuleDependency(ModuleDeps MD) override {}

  void handleVisibleModule(std::string ModuleName) override {}

  void handleDirectModuleDependency(ModuleID ID) override {}

  void handleContextHash(std::string Hash) override {}
};

/// Returns an IncludeTree containing the dependencies.
class GetIncludeTree : public EmptyDependencyConsumer {
public:
  void handleIncludeTreeID(std::string ID) override { IncludeTreeID = ID; }

  Expected<cas::IncludeTreeRoot> getIncludeTree() {
    if (IncludeTreeID) {
      auto ID = DB.parseID(*IncludeTreeID);
      if (!ID)
        return ID.takeError();
      auto Ref = DB.getReference(*ID);
      if (!Ref)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            llvm::Twine("missing expected include-tree ") + ID->toString());
      return cas::IncludeTreeRoot::get(DB, *Ref);
    }
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "failed to get include-tree");
  }

  GetIncludeTree(cas::ObjectStore &DB) : DB(DB) {}

private:
  cas::ObjectStore &DB;
  std::optional<std::string> IncludeTreeID;
};

struct AlreadyReportedDiagnosticError
    : llvm::ErrorInfo<AlreadyReportedDiagnosticError> {
  static char ID;

  void log(raw_ostream &OS) const override {
    OS << "failed to scan dependencies";
  }

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }
};

char AlreadyReportedDiagnosticError::ID = 0;
} // namespace

Expected<cas::IncludeTreeRoot> DependencyScanningTool::getIncludeTree(
    const std::vector<std::string> &CommandLine, StringRef CWD,
    LookupModuleOutputCallback LookupModuleOutput,
    DiagnosticConsumer &DiagsConsumer) {
  GetIncludeTree Consumer(*getCAS());
  auto Controller = createIncludeTreeActionController(LookupModuleOutput,
                                                      getCASOpts(), *getCAS());
  if (!computeDependencies(Worker, CWD, CommandLine, Consumer, *Controller,
                           DiagsConsumer))
    return llvm::make_error<AlreadyReportedDiagnosticError>();
  return Consumer.getIncludeTree();
}

Expected<cas::IncludeTreeRoot>
DependencyScanningTool::getIncludeTreeFromCompilerInvocation(
    std::shared_ptr<CompilerInvocation> Invocation, StringRef CWD,
    LookupModuleOutputCallback LookupModuleOutput,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS) {
  GetIncludeTree Consumer(*getCAS());
  auto Controller = createIncludeTreeActionController(LookupModuleOutput,
                                                      getCASOpts(), *getCAS());
  Worker.computeDependenciesFromCompilerInvocation(std::move(Invocation), CWD,
                                                   Consumer, *Controller,
                                                   DiagsConsumer, VerboseOS);
  return Consumer.getIncludeTree();
}

std::optional<P1689Rule> DependencyScanningTool::getP1689ModuleDependencyFile(
    const CompileCommand &Command, StringRef CWD, std::string &MakeformatOutput,
    std::string &MakeformatOutputPath, DiagnosticConsumer &DiagConsumer) {
  class P1689ModuleDependencyPrinterConsumer
      : public MakeDependencyPrinterConsumer {
  public:
    P1689ModuleDependencyPrinterConsumer(P1689Rule &Rule,
                                         const CompileCommand &Command)
        : Filename(Command.Filename), Rule(Rule) {
      Rule.PrimaryOutput = Command.Output;
    }

    void handleProvidedAndRequiredStdCXXModules(
        std::optional<P1689ModuleInfo> Provided,
        std::vector<P1689ModuleInfo> Requires) override {
      Rule.Provides = std::move(Provided);
      if (Rule.Provides)
        Rule.Provides->SourcePath = Filename.str();
      Rule.Requires = std::move(Requires);
    }

    StringRef getMakeFormatDependencyOutputPath() {
      if (Opts->OutputFormat != DependencyOutputFormat::Make)
        return {};
      return Opts->OutputFile;
    }

  private:
    StringRef Filename;
    P1689Rule &Rule;
  };

  class P1689ActionController : public DependencyActionController {
  public:
    // The lookupModuleOutput is for clang modules. P1689 format don't need it.
    std::string lookupModuleOutput(const ModuleDeps &,
                                   ModuleOutputKind Kind) override {
      return "";
    }

    std::unique_ptr<DependencyActionController> clone() const override {
      return std::make_unique<P1689ActionController>();
    }
  };

  P1689Rule Rule;
  P1689ModuleDependencyPrinterConsumer Consumer(Rule, Command);
  P1689ActionController Controller;
  if (!computeDependencies(Worker, CWD, Command.CommandLine, Consumer,
                           Controller, DiagConsumer))
    return std::nullopt;

  MakeformatOutputPath = Consumer.getMakeFormatDependencyOutputPath();
  if (!MakeformatOutputPath.empty())
    Consumer.printDependencies(MakeformatOutput);
  return Rule;
}

static std::pair<IntrusiveRefCntPtr<llvm::vfs::FileSystem>,
                 std::vector<std::string>>
initVFSForTUBufferScanning(ArrayRef<std::string> CommandLine,
                           llvm::MemoryBufferRef TUBuffer) {
  StringRef InputPath = TUBuffer.getBufferIdentifier();
  auto InputBuf = llvm::MemoryBuffer::getMemBufferCopy(TUBuffer.getBuffer());

  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FS->addFile(InputPath, 0, std::move(InputBuf));

  std::vector<std::string> ModifiedCommandLine(CommandLine);
  ModifiedCommandLine.emplace_back(InputPath);

  return std::make_pair(std::move(FS), ModifiedCommandLine);
}

static std::pair<IntrusiveRefCntPtr<llvm::vfs::FileSystem>,
                 std::vector<std::string>>
initVFSForByNameScanning(ArrayRef<std::string> CommandLine) {
  // The fake input buffer is read-only, and it is used to produce unique source
  // locations for the diagnostics. Therefore, sharing this global buffer across
  // threads is ok.
  static const std::string FakeInput(
      CompilerInstanceWithContext::MaxNumOfQueries, ' ');

  StringRef InputPath =
      llvm::sys::path::is_style_windows(llvm::sys::path::Style::native)
          ? "Z:\\module-include.input"
          : "/module-include.input";
  auto InputBuf = llvm::MemoryBuffer::getMemBuffer(FakeInput, InputPath);

  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FS->addFile(InputPath, 0, std::move(InputBuf));

  std::vector<std::string> ModifiedCommandLine(CommandLine);
  ModifiedCommandLine.emplace_back(InputPath);

  return std::make_pair(std::move(FS), ModifiedCommandLine);
}

std::optional<TranslationUnitDeps>
DependencyScanningTool::getTranslationUnitDependencies(
    ArrayRef<std::string> CommandLine, StringRef CWD,
    DiagnosticConsumer &DiagConsumer,
    const llvm::DenseSet<ModuleID> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput,
    std::optional<llvm::MemoryBufferRef> TUBuffer) {
  FullDependencyConsumer Consumer(AlreadySeen);
  auto Controller = createActionController(LookupModuleOutput);

  // If we are scanning from a TUBuffer, create an overlay filesystem with the
  // input as an in-memory file and add it to the command line.
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS = nullptr;
  std::vector<std::string> CommandLineWithTUBufferInput;
  if (TUBuffer) {
    std::tie(OverlayFS, CommandLineWithTUBufferInput) =
        initVFSForTUBufferScanning(CommandLine, *TUBuffer);
    CommandLine = CommandLineWithTUBufferInput;
  }

  if (!computeDependencies(Worker, CWD, CommandLine, Consumer, *Controller,
                           DiagConsumer, std::move(OverlayFS)))
    return std::nullopt;
  return Consumer.takeTranslationUnitDeps();
}

llvm::Expected<TranslationUnitDeps>
DependencyScanningTool::getModuleDependencies(
    StringRef ModuleName, ArrayRef<std::string> CommandLine, StringRef CWD,
    const llvm::DenseSet<ModuleID> &AlreadySeen,
    DependencyActionController &Controller) {
  auto MaybeCIWithContext = CompilerInstanceWithContext::initializeOrError(
      *this, CWD, CommandLine, Controller);
  if (auto Error = MaybeCIWithContext.takeError())
    return Error;

  return MaybeCIWithContext->computeDependenciesByNameOrError(
      ModuleName, AlreadySeen, Controller);
}

static std::optional<SmallVector<std::string, 0>>
getFirstCC1CommandLine(ArrayRef<std::string> CommandLine,
                       DiagnosticsEngine &Diags,
                       llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) {
  // Compilation holds a non-owning a reference to the Driver, hence we need to
  // keep the Driver alive when we use Compilation. Arguments to commands may be
  // owned by Alloc when expanded from response files.
  llvm::BumpPtrAllocator Alloc;
  const auto [Driver, Compilation] =
      buildCompilation(CommandLine, Diags, std::move(FS), Alloc);
  if (!Compilation)
    return std::nullopt;

  const auto IsClangCmd = [](const driver::Command &Cmd) {
    return StringRef(Cmd.getCreator().getName()) == "clang";
  };

  const auto &Jobs = Compilation->getJobs();
  if (const auto It = llvm::find_if(Jobs, IsClangCmd); It != Jobs.end())
    return buildCC1CommandLine(*It);
  return std::nullopt;
}

std::optional<CompilerInstanceWithContext>
CompilerInstanceWithContext::initializeFromCommandline(
    DependencyScanningTool &Tool, StringRef CWD,
    ArrayRef<std::string> CommandLine, DependencyActionController &Controller,
    DiagnosticConsumer &DC) {
  auto [OverlayFS, ModifiedCommandLine] = initVFSForByNameScanning(CommandLine);
  auto FS = Tool.Worker.makeEffectiveVFS(CWD, OverlayFS);

  auto DiagEngineWithCmdAndOpts =
      std::make_unique<DiagnosticsEngineWithDiagOpts>(ModifiedCommandLine, FS,
                                                      DC);

  if (ModifiedCommandLine.size() >= 2 && ModifiedCommandLine[1] == "-cc1") {
    // The input command line is already a -cc1 invocation; initialize the
    // compiler instance directly from it.
    CompilerInstanceWithContext CIWithContext(Tool.Worker, CWD,
                                              ModifiedCommandLine);
    if (!CIWithContext.initialize(Controller,
                                  std::move(DiagEngineWithCmdAndOpts),
                                  std::move(OverlayFS)))
      return std::nullopt;
    return std::move(CIWithContext);
  }

  // The input command line is either a driver-style command line, or
  // ill-formed. In this case, we will first call the Driver to build a -cc1
  // command line for this compilation or diagnose any ill-formed input.
  const auto MaybeFirstCC1 = getFirstCC1CommandLine(
      ModifiedCommandLine, *DiagEngineWithCmdAndOpts->DiagEngine, FS);
  if (!MaybeFirstCC1)
    return std::nullopt;

  std::vector<std::string> CC1CommandLine(MaybeFirstCC1->begin(),
                                          MaybeFirstCC1->end());
  CompilerInstanceWithContext CIWithContext(Tool.Worker, CWD,
                                            std::move(CC1CommandLine));
  if (!CIWithContext.initialize(Controller, std::move(DiagEngineWithCmdAndOpts),
                                std::move(OverlayFS)))
    return std::nullopt;
  return std::move(CIWithContext);
}

llvm::Expected<CompilerInstanceWithContext>
CompilerInstanceWithContext::initializeOrError(
    DependencyScanningTool &Tool, StringRef CWD,
    ArrayRef<std::string> CommandLine, DependencyActionController &Controller) {
  auto DiagPrinterWithOS =
      std::make_unique<TextDiagnosticsPrinterWithOutput>(CommandLine);

  auto Result = initializeFromCommandline(Tool, CWD, CommandLine, Controller,
                                          DiagPrinterWithOS->DiagPrinter);
  if (Result) {
    Result->DiagPrinterWithOS = std::move(DiagPrinterWithOS);
    return std::move(*Result);
  }
  return makeErrorFromDiagnosticsOS(*DiagPrinterWithOS);
}

llvm::Expected<TranslationUnitDeps>
CompilerInstanceWithContext::computeDependenciesByNameOrError(
    StringRef ModuleName, const llvm::DenseSet<ModuleID> &AlreadySeen,
    DependencyActionController &Controller) {
  // FIXME: Make IncludeTreeActionController re-entrant and avoid cloning here.
  auto ControllerClone = Controller.clone();
  FullDependencyConsumer Consumer(AlreadySeen);
  // We need to clear the DiagnosticOutput so that each by-name lookup
  // has a clean diagnostics buffer.
  DiagPrinterWithOS->DiagnosticOutput.clear();
  if (computeDependencies(ModuleName, Consumer, *ControllerClone))
    return Consumer.takeTranslationUnitDeps();
  return makeErrorFromDiagnosticsOS(*DiagPrinterWithOS);
}

std::unique_ptr<DependencyActionController>
DependencyScanningTool::createActionController(
    DependencyScanningWorker &Worker,
    LookupModuleOutputCallback LookupModuleOutput) {
  if (auto *IncludeTree = std::get_if<IncludeTreeCompilation>(
          &Worker.getService().getOpts().Compilation))
    return createIncludeTreeActionController(
        LookupModuleOutput, IncludeTree->CASOpts, *IncludeTree->CAS);
  return std::make_unique<CallbackActionController>(LookupModuleOutput);
}

std::unique_ptr<DependencyActionController>
DependencyScanningTool::createActionController(
    LookupModuleOutputCallback LookupModuleOutput) {
  return createActionController(Worker, std::move(LookupModuleOutput));
}

Expected<llvm::cas::CASID> clang::scanAndUpdateCC1InlineWithTool(
    DependencyScanningTool &Tool, DiagnosticConsumer &DiagsConsumer,
    raw_ostream *VerboseOS, CompilerInvocation &Invocation,
    StringRef WorkingDirectory) {
  llvm::PrefixMapper Mapper;
  DepscanPrefixMapping::configurePrefixMapper(Invocation, Mapper);

  auto ScanInvocation = std::make_shared<CompilerInvocation>(Invocation);
  // An error during dep-scanning is treated as if the main compilation has
  // failed, but warnings are ignored and deferred for the main compilation.
  ScanInvocation->getDiagnosticOpts().IgnoreWarnings = true;

  // Make the output file path absolute relative to WorkingDirectory.
  std::string &DepFile = ScanInvocation->getDependencyOutputOpts().OutputFile;
  if (!DepFile.empty() && !llvm::sys::path::is_absolute(DepFile)) {
    // FIXME: On Windows, WorkingDirectory is insufficient for making an
    // absolute path if OutputFile has a root name.
    llvm::SmallString<128> Path = StringRef(DepFile);
    llvm::sys::path::make_absolute(WorkingDirectory, Path);
    DepFile = Path.str().str();
  }

  std::optional<llvm::cas::CASID> Root;
  if (Error E =
          Tool.getIncludeTreeFromCompilerInvocation(
                  std::move(ScanInvocation), WorkingDirectory,
                  /*LookupModuleOutput=*/nullptr, DiagsConsumer, VerboseOS)
              .moveInto(Root))
    return std::move(E);

  // Turn off dependency outputs. Should have already been emitted.
  Invocation.getDependencyOutputOpts().OutputFile.clear();

  configureInvocationForCaching(Invocation, Tool.getCASOpts(), Root->toString(),
                                CachingInputKind::IncludeTree,
                                WorkingDirectory.str());
  DepscanPrefixMapping::remapInvocationPaths(Invocation, Mapper);
  return *Root;
}

bool CompilerInstanceWithContext::initialize(
    DependencyActionController &Controller,
    std::unique_ptr<DiagnosticsEngineWithDiagOpts> DiagEngineWithDiagOpts,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS) {
  assert(DiagEngineWithDiagOpts && "Valid diagnostics engine required!");
  DiagEngineWithCmdAndOpts = std::move(DiagEngineWithDiagOpts);
  DiagConsumer = DiagEngineWithCmdAndOpts->DiagEngine->getClient();

  assert(OverlayFS && "OverlayFS required!");
  auto FS = Worker.makeEffectiveVFS(CWD, std::move(OverlayFS));

  OriginalInvocation = createCompilerInvocation(
      CommandLine, *DiagEngineWithCmdAndOpts->DiagEngine);
  if (!OriginalInvocation) {
    DiagEngineWithCmdAndOpts->DiagEngine->Report(
        diag::err_fe_expected_compiler_job)
        << llvm::join(CommandLine, " ");
    return false;
  }

  if (any(Worker.Service.getOpts().OptimizeArgs &
          ScanningOptimizations::Macros))
    canonicalizeDefines(OriginalInvocation->getPreprocessorOpts());

  // Create the CompilerInstance.
  std::shared_ptr<ModuleCache> ModCache =
      makeInProcessModuleCache(Worker.Service.getModuleCacheEntries());
  CIPtr = std::make_unique<CompilerInstance>(
      createScanCompilerInvocation(*OriginalInvocation, Worker.Service,
                                   Controller),
      Worker.PCHContainerOps, std::move(ModCache));
  auto &CI = *CIPtr;

  initializeScanCompilerInstance(
      CI, std::move(FS), DiagEngineWithCmdAndOpts->DiagEngine->getClient(),
      Worker.Service, Worker.DepFS);

  StableDirs = getInitialStableDirs(CI);
  auto MaybePrebuiltModulesASTMap =
      computePrebuiltModulesASTMap(CI, StableDirs);
  if (!MaybePrebuiltModulesASTMap)
    return false;

  PrebuiltModuleASTMap = std::move(*MaybePrebuiltModulesASTMap);
  // FIXME: Set ForceIncludeSystemHeaders for Make consumers.
  OutputOpts =
      createDependencyOutputOptions(*OriginalInvocation,
                                    /*ForceIncludeSystemHeaders=*/false);

  // We do not create the target in initializeScanCompilerInstance because
  // setting it here is unique for by-name lookups. We create the target only
  // once here, and the information is reused for all computeDependencies calls.
  // We do not need to call createTarget explicitly if we go through
  // CompilerInstance::ExecuteAction to perform scanning.
  CI.createTarget();
  CI.initializeDelayedInputFileFromCAS();

  return true;
}

bool CompilerInstanceWithContext::computeDependencies(
    StringRef ModuleName, DependencyConsumer &Consumer,
    DependencyActionController &Controller) {
  if (SrcLocOffset >= MaxNumOfQueries)
    llvm::report_fatal_error("exceeded maximum by-name scans for worker");

  assert(CIPtr && "CIPtr must be initialized before calling this method");
  auto &CI = *CIPtr;

  // We need to reset the diagnostics, so that the diagnostics issued
  // during a previous computeDependencies call do not affect the current call.
  // If we do not reset, we may inherit fatal errors from a previous call.
  CI.getDiagnostics().Reset();

  // We create this cleanup object because computeDependencies may exit
  // early with errors.
  llvm::scope_exit CleanUp([&]() {
    CI.clearDependencyCollectors();

    // Clean up the PPCallbacks if we have a preprocessor setup.
    if (CI.hasPreprocessor())
      CI.getPreprocessor().removePPCallbacks();
  });

  auto MDC = initializeScanInstanceDependencyCollector(
      CI, std::make_unique<DependencyOutputOptions>(*OutputOpts),
      Worker.Service,
      /* The MDC's constructor makes a copy of the OriginalInvocation, so
      we can pass it in without worrying that it might be changed across
      invocations of computeDependencies. */
      *OriginalInvocation, Controller, PrebuiltModuleASTMap, StableDirs);

  CompilerInvocation ModuleInvocation(*OriginalInvocation);
  if (!Controller.initialize(CI, ModuleInvocation))
    return false;

  if (!SrcLocOffset) {
    // When SrcLocOffset is zero, we are at the beginning of the fake source
    // file. In this case, we call BeginSourceFile to initialize.
    std::unique_ptr<FrontendAction> Action =
        std::make_unique<PreprocessOnlyAction>();
    auto *InputFile = CI.getFrontendOpts().Inputs.begin();
    bool ActionBeginSucceeded = Action->BeginSourceFile(CI, *InputFile);
    assert(ActionBeginSucceeded && "Action BeginSourceFile must succeed");
    (void)ActionBeginSucceeded;
  }

  Preprocessor &PP = CI.getPreprocessor();
  SourceManager &SM = PP.getSourceManager();
  FileID MainFileID = SM.getMainFileID();
  SourceLocation FileStart = SM.getLocForStartOfFile(MainFileID);
  SourceLocation IDLocation = FileStart.getLocWithOffset(SrcLocOffset);
  PPCallbacks *CB = nullptr;
  if (!SrcLocOffset) {
    // We need to call EnterSourceFile when SrcLocOffset is zero to initialize
    // the preprocessor.
    bool PPFailed = PP.EnterSourceFile(MainFileID, nullptr, SourceLocation());
    assert(!PPFailed && "Preprocess must be able to enter the main file.");
    (void)PPFailed;
    CB = PP.getPPCallbacks();
  } else {
    // When SrcLocOffset is non-zero, the preprocessor has already been
    // initialized through a previous call of computeDependencies. We want to
    // preserve the PP's state, hence we do not call EnterSourceFile again.
    auto DCs = CI.getDependencyCollectors();
    for (auto &DC : DCs)
      DC->attachToPreprocessor(PP);

    CB = PP.getPPCallbacks();
    FileID PrevFID;
    SrcMgr::CharacteristicKind FileType = SM.getFileCharacteristic(IDLocation);
    CB->LexedFileChanged(MainFileID,
                         PPChainedCallbacks::LexedFileChangeReason::EnterFile,
                         FileType, PrevFID, IDLocation);
  }

  // FIXME: Scan modules asynchronously here as well.

  SrcLocOffset++;
  SmallVector<IdentifierLoc, 2> Path;
  IdentifierInfo *ModuleID = PP.getIdentifierInfo(ModuleName);
  Path.emplace_back(IDLocation, ModuleID);
  auto ModResult = CI.loadModule(IDLocation, Path, Module::Hidden, false);

  assert(CB && "Must have PPCallbacks after module loading");
  CB->moduleImport(SourceLocation(), Path, ModResult);

  if (!ModResult)
    return false;

  if (CI.getDiagnostics().hasErrorOccurred())
    return false;

  MDC->run(Consumer);
  MDC->applyDiscoveredDependencies(ModuleInvocation);

  if (!Controller.finalize(CI, ModuleInvocation))
    return false;

  std::string ID = ModuleInvocation.getFrontendOpts().CASIncludeTreeID;
  if (!ID.empty())
    Consumer.handleIncludeTreeID(std::move(ID));

  auto LastCC1Arguments = ModuleInvocation.getCC1CommandLine();
  auto LastCC1CacheKey = Controller.getCacheKey(ModuleInvocation);

  Consumer.handleBuildCommand({CommandLine[0], std::move(LastCC1Arguments),
                               std::move(LastCC1CacheKey)});

  return true;
}
