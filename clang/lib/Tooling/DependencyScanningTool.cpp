//===- DependencyScanningTool.cpp - clang-scan-deps service ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanningTool.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/CAS/IncludeTree.h"
#include "clang/DependencyScanning/CachingActions.h"
#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/Utils.h"
#include "llvm/CAS/ObjectStore.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;
using llvm::Error;

DependencyScanningTool::DependencyScanningTool(
    DependencyScanningService &Service,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : Worker(Service, std::move(FS)) {}

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
    Dependencies.push_back(std::string(File));
  }

  // These are ignored for the make format as it can't support the full
  // set of deps, and handleFileDependency handles enough for implicitly
  // built modules to work.
  void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) override {}
  void handleModuleDependency(ModuleDeps MD) override {}
  void handleDirectModuleDependency(ModuleID ID) override {}
  void handleVisibleModule(std::string ModuleName) override {}
  void handleContextHash(std::string Hash) override {}

  void printDependencies(std::string &S) {
    assert(Opts && "Handled dependency output options.");

    class DependencyPrinter : public DependencyFileGenerator {
    public:
      DependencyPrinter(DependencyOutputOptions &Opts,
                        ArrayRef<std::string> Dependencies)
          : DependencyFileGenerator(Opts) {
        for (const auto &Dep : Dependencies)
          addDependency(Dep);
      }

      void printDependencies(std::string &S) {
        llvm::raw_string_ostream OS(S);
        outputDependencyFile(OS);
      }
    };

    DependencyPrinter Generator(*Opts, Dependencies);
    Generator.printDependencies(S);
  }

protected:
  std::unique_ptr<DependencyOutputOptions> Opts;
  std::vector<std::string> Dependencies;
};
} // anonymous namespace

llvm::Expected<std::string>
DependencyScanningTool::getDependencyFile(ArrayRef<std::string> CommandLine,
                                          StringRef CWD) {
  MakeDependencyPrinterConsumer Consumer;
  CallbackActionController Controller(nullptr);
  auto Result =
      Worker.computeDependencies(CWD, CommandLine, Consumer, Controller);
  if (Result)
    return std::move(Result);
  std::string Output;
  Consumer.printDependencies(Output);
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
} // namespace

Expected<cas::IncludeTreeRoot> DependencyScanningTool::getIncludeTree(
    cas::ObjectStore &DB, const std::vector<std::string> &CommandLine,
    StringRef CWD, LookupModuleOutputCallback LookupModuleOutput) {
  GetIncludeTree Consumer(DB);
  auto Controller = createIncludeTreeActionController(LookupModuleOutput, DB);
  llvm::Error Result =
      Worker.computeDependencies(CWD, CommandLine, Consumer, *Controller);
  if (Result)
    return std::move(Result);
  return Consumer.getIncludeTree();
}

Expected<cas::IncludeTreeRoot>
DependencyScanningTool::getIncludeTreeFromCompilerInvocation(
    cas::ObjectStore &DB, std::shared_ptr<CompilerInvocation> Invocation,
    StringRef CWD, LookupModuleOutputCallback LookupModuleOutput,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS,
    bool DiagGenerationAsCompilation) {
  GetIncludeTree Consumer(DB);
  auto Controller = createIncludeTreeActionController(LookupModuleOutput, DB);
  Worker.computeDependenciesFromCompilerInvocation(
      std::move(Invocation), CWD, Consumer, *Controller, DiagsConsumer,
      VerboseOS, DiagGenerationAsCompilation);
  return Consumer.getIncludeTree();
}

llvm::Expected<P1689Rule> DependencyScanningTool::getP1689ModuleDependencyFile(
    const CompileCommand &Command, StringRef CWD, std::string &MakeformatOutput,
    std::string &MakeformatOutputPath) {
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
      Rule.Provides = Provided;
      if (Rule.Provides)
        Rule.Provides->SourcePath = Filename.str();
      Rule.Requires = Requires;
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
  };

  P1689Rule Rule;
  P1689ModuleDependencyPrinterConsumer Consumer(Rule, Command);
  P1689ActionController Controller;
  auto Result = Worker.computeDependencies(CWD, Command.CommandLine, Consumer,
                                           Controller);
  if (Result)
    return std::move(Result);

  MakeformatOutputPath = Consumer.getMakeFormatDependencyOutputPath();
  if (!MakeformatOutputPath.empty())
    Consumer.printDependencies(MakeformatOutput);
  return Rule;
}

llvm::Expected<TranslationUnitDeps>
DependencyScanningTool::getTranslationUnitDependencies(
    ArrayRef<std::string> CommandLine, StringRef CWD,
    const llvm::DenseSet<ModuleID> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput,
    std::optional<llvm::MemoryBufferRef> TUBuffer) {
  FullDependencyConsumer Consumer(AlreadySeen);
  auto Controller = createActionController(LookupModuleOutput);
  llvm::Error Result = Worker.computeDependencies(CWD, CommandLine, Consumer,
                                                  *Controller, TUBuffer);
  if (Result)
    return std::move(Result);
  return Consumer.takeTranslationUnitDeps();
}

llvm::Expected<TranslationUnitDeps>
DependencyScanningTool::getModuleDependencies(
    StringRef ModuleName, ArrayRef<std::string> CommandLine, StringRef CWD,
    const llvm::DenseSet<ModuleID> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput) {
  if (auto Error =
          initializeCompilerInstanceWithContextOrError(CWD, CommandLine))
    return Error;

  auto Result = computeDependenciesByNameWithContextOrError(
      ModuleName, AlreadySeen, LookupModuleOutput);

  if (auto Error = finalizeCompilerInstanceWithContextOrError())
    return Error;

  return Result;
}

/// Constructs the full -cc1 command line, including executable, for the given
/// driver \c Cmd.
static std::vector<std::string>
buildCC1CommandLine(const driver::Command &Cmd) {
  const auto &Args = Cmd.getArguments();
  std::vector<std::string> Out;
  Out.reserve(Args.size() + 1);
  Out.emplace_back(Cmd.getExecutable());
  llvm::append_range(Out, Args);
  return Out;
}

std::optional<std::vector<std::string>> tooling::getFirstCC1CommandLine(
    ArrayRef<std::string> CommandLine, DiagnosticsEngine &Diags,
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> ScanFS) {
  // Compilation holds a non-owning a reference to the Driver, hence we need to
  // keep the Driver alive when we use Compilation. Arguments to commands may be
  // owned by Alloc when expanded from response files.
  llvm::BumpPtrAllocator Alloc;
  const auto [Driver, Compilation] =
      buildCompilation(CommandLine, Diags, ScanFS, Alloc);
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

static llvm::Error makeErrorFromDiagnosticsOS(
    TextDiagnosticsPrinterWithOutput &DiagPrinterWithOS) {
  return llvm::make_error<llvm::StringError>(
      DiagPrinterWithOS.DiagnosticsOS.str(), llvm::inconvertibleErrorCode());
}

bool DependencyScanningTool::initializeWorkerCIWithContextFromCommandline(
    DependencyScanningWorker &Worker, StringRef CWD,
    ArrayRef<std::string> CommandLine, DiagnosticConsumer &DC) {
  if (CommandLine.size() >= 2 && CommandLine[1] == "-cc1") {
    // The input command line is already a -cc1 invocation; initialize the
    // compiler instance directly from it.
    return Worker.initializeCompilerInstanceWithContext(CWD, CommandLine, DC);
  }

  // The input command line is either a driver-style command line, or
  // ill-formed. In this case, we will first call the Driver to build a -cc1
  // command line for this compilation or diagnose any ill-formed input.
  auto [OverlayFS, ModifiedCommandLine] = initVFSForByNameScanning(
      &Worker.getVFS(), CommandLine, CWD, "ScanningByName", Worker.getCAS());
  auto DiagEngineWithCmdAndOpts =
      std::make_unique<DiagnosticsEngineWithDiagOpts>(ModifiedCommandLine,
                                                      OverlayFS, DC);

  const auto MaybeFirstCC1 = getFirstCC1CommandLine(
      ModifiedCommandLine, *DiagEngineWithCmdAndOpts->DiagEngine, OverlayFS);
  if (!MaybeFirstCC1)
    return false;

  return Worker.initializeCompilerInstanceWithContext(
      CWD, *MaybeFirstCC1, std::move(DiagEngineWithCmdAndOpts), OverlayFS);
}

llvm::Error
DependencyScanningTool::initializeCompilerInstanceWithContextOrError(
    StringRef CWD, ArrayRef<std::string> CommandLine) {
  DiagPrinterWithOS =
      std::make_unique<TextDiagnosticsPrinterWithOutput>(CommandLine);

  bool Result = initializeWorkerCIWithContextFromCommandline(
      Worker, CWD, CommandLine, DiagPrinterWithOS->DiagPrinter);

  if (Result)
    return llvm::Error::success();
  else
    return makeErrorFromDiagnosticsOS(*DiagPrinterWithOS);
}

llvm::Expected<TranslationUnitDeps>
DependencyScanningTool::computeDependenciesByNameWithContextOrError(
    StringRef ModuleName, const llvm::DenseSet<ModuleID> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput) {
  FullDependencyConsumer Consumer(AlreadySeen);
  auto Controller = createActionController(LookupModuleOutput);
  if (Worker.computeDependenciesByNameWithContext(ModuleName, Consumer,
                                                  *Controller))
    return Consumer.takeTranslationUnitDeps();
  return makeErrorFromDiagnosticsOS(*DiagPrinterWithOS);
}

llvm::Error
DependencyScanningTool::finalizeCompilerInstanceWithContextOrError() {
  if (Worker.finalizeCompilerInstanceWithContext())
    return llvm::Error::success();
  return makeErrorFromDiagnosticsOS(*DiagPrinterWithOS);
}

std::unique_ptr<DependencyActionController>
DependencyScanningTool::createActionController(
    DependencyScanningWorker &Worker,
    LookupModuleOutputCallback LookupModuleOutput) {
  if (Worker.getScanningFormat() == ScanningOutputFormat::FullIncludeTree)
    return createIncludeTreeActionController(LookupModuleOutput,
                                             *Worker.getCAS());
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
    StringRef WorkingDirectory, llvm::cas::ObjectStore &DB) {
  // Override the CASOptions. They may match (the caller having sniffed them
  // out of InputArgs) but if they have been overridden we want the new ones.
  Invocation.getCASOpts() = Tool.getCASOpts();

  llvm::PrefixMapper Mapper;
  DepscanPrefixMapping::configurePrefixMapper(Invocation, Mapper);

  auto ScanInvocation = std::make_shared<CompilerInvocation>(Invocation);
  // An error during dep-scanning is treated as if the main compilation has
  // failed, but warnings are ignored and deferred for the main compilation.
  ScanInvocation->getDiagnosticOpts().IgnoreWarnings = true;

  LookupModuleOutputCallback Lookup;

  std::optional<llvm::cas::CASID> Root;
  if (Error E =
          Tool.getIncludeTreeFromCompilerInvocation(
                  DB, std::move(ScanInvocation), WorkingDirectory,
                  /*LookupModuleOutput=*/nullptr, DiagsConsumer, VerboseOS,
                  /*DiagGenerationAsCompilation*/ true)
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
