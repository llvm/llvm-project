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
#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/Utils.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/iterator.h"
#include "llvm/TargetParser/Host.h"
#include <optional>

using namespace clang;
using namespace tooling;
using namespace dependencies;

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
    IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS) {
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS = nullptr;
  if (OverlayFS) {
    FS = OverlayFS;
  } else {
    FS = &Worker.getVFS();
    FS->setCurrentWorkingDirectory(WorkingDirectory);
  }

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
                                    OverlayFS);
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
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS) {
  const auto IsCC1Input = (CommandLine.size() >= 2 && CommandLine[1] == "-cc1");
  return IsCC1Input ? Worker.computeDependencies(WorkingDirectory, CommandLine,
                                                 Consumer, Controller,
                                                 DiagConsumer, OverlayFS)
                    : computeDependenciesForDriverCommandLine(
                          Worker, WorkingDirectory, CommandLine, Consumer,
                          Controller, DiagConsumer, OverlayFS);
}

std::optional<std::string>
DependencyScanningTool::getDependencyFile(ArrayRef<std::string> CommandLine,
                                          StringRef CWD,
                                          DiagnosticConsumer &DiagConsumer) {
  MakeDependencyPrinterConsumer DepConsumer;
  CallbackActionController Controller(nullptr);
  if (!computeDependencies(Worker, CWD, CommandLine, DepConsumer, Controller,
                           DiagConsumer))
    return std::nullopt;
  std::string Output;
  DepConsumer.printDependencies(Output);
  return Output;
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
      Rule.Provides = Provided;
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

std::optional<TranslationUnitDeps>
DependencyScanningTool::getTranslationUnitDependencies(
    ArrayRef<std::string> CommandLine, StringRef CWD,
    DiagnosticConsumer &DiagConsumer,
    const llvm::DenseSet<ModuleID> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput,
    std::optional<llvm::MemoryBufferRef> TUBuffer) {
  FullDependencyConsumer Consumer(AlreadySeen);
  CallbackActionController Controller(LookupModuleOutput);

  // If we are scanning from a TUBuffer, create an overlay filesystem with the
  // input as an in-memory file and add it to the command line.
  IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS = nullptr;
  std::vector<std::string> CommandLineWithTUBufferInput;
  if (TUBuffer) {
    std::tie(OverlayFS, CommandLineWithTUBufferInput) =
        initVFSForTUBufferScanning(&Worker.getVFS(), CommandLine, CWD,
                                   *TUBuffer);
    CommandLine = CommandLineWithTUBufferInput;
  }

  if (!computeDependencies(Worker, CWD, CommandLine, Consumer, Controller,
                           DiagConsumer, OverlayFS))
    return std::nullopt;
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

static std::optional<SmallVector<std::string, 0>> getFirstCC1CommandLine(
    ArrayRef<std::string> CommandLine, DiagnosticsEngine &Diags,
    llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> OverlayFS) {
  // Compilation holds a non-owning a reference to the Driver, hence we need to
  // keep the Driver alive when we use Compilation. Arguments to commands may be
  // owned by Alloc when expanded from response files.
  llvm::BumpPtrAllocator Alloc;
  const auto [Driver, Compilation] =
      buildCompilation(CommandLine, Diags, OverlayFS, Alloc);
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
      &Worker.getVFS(), CommandLine, CWD, "ScanningByName");
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
  return makeErrorFromDiagnosticsOS(*DiagPrinterWithOS);
}

llvm::Expected<TranslationUnitDeps>
DependencyScanningTool::computeDependenciesByNameWithContextOrError(
    StringRef ModuleName, const llvm::DenseSet<ModuleID> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput) {
  FullDependencyConsumer Consumer(AlreadySeen);
  CallbackActionController Controller(LookupModuleOutput);
  // We need to clear the DiagnosticOutput so that each by-name lookup
  // has a clean diagnostics buffer.
  DiagPrinterWithOS->DiagnosticOutput.clear();
  if (Worker.computeDependenciesByNameWithContext(ModuleName, Consumer,
                                                  Controller))
    return Consumer.takeTranslationUnitDeps();
  return makeErrorFromDiagnosticsOS(*DiagPrinterWithOS);
}

llvm::Error
DependencyScanningTool::finalizeCompilerInstanceWithContextOrError() {
  if (Worker.finalizeCompilerInstanceWithContext())
    return llvm::Error::success();
  return makeErrorFromDiagnosticsOS(*DiagPrinterWithOS);
}
