//===- DependencyScanningTool.cpp - clang-scan-deps service ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanningTool.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/Utils.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/TargetParser/Host.h"
#include <optional>

using namespace clang;
using namespace tooling;
using namespace clang::dependencies;
using namespace clang::tooling::dependencies;

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

static std::pair<std::unique_ptr<driver::Driver>,
                 std::unique_ptr<driver::Compilation>>
buildCompilation(ArrayRef<std::string> CommandLine, DiagnosticsEngine &Diags,
                 IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS,
                 llvm::BumpPtrAllocator &Alloc) {
  SmallVector<const char *, 256> Argv;
  Argv.reserve(CommandLine.size());
  for (const std::string &Arg : CommandLine)
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
        << llvm::join(CommandLine, " ");
    return std::make_pair(nullptr, nullptr);
  }

  return std::make_pair(std::move(Driver), std::move(Compilation));
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

static std::pair<IntrusiveRefCntPtr<llvm::vfs::FileSystem>,
                 std::vector<std::string>>
initVFSForTUBufferScanning(IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
                           ArrayRef<std::string> CommandLine,
                           StringRef WorkingDirectory,
                           llvm::MemoryBufferRef TUBuffer) {
  // Reset what might have been modified in the previous worker invocation.
  BaseFS->setCurrentWorkingDirectory(WorkingDirectory);

  IntrusiveRefCntPtr<llvm::vfs::FileSystem> ModifiedFS;
  auto OverlayFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(BaseFS);
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory(WorkingDirectory);
  auto InputPath = TUBuffer.getBufferIdentifier();
  InMemoryFS->addFile(
      InputPath, 0, llvm::MemoryBuffer::getMemBufferCopy(TUBuffer.getBuffer()));
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> InMemoryOverlay = InMemoryFS;

  OverlayFS->pushOverlay(InMemoryOverlay);
  std::vector<std::string> ModifiedCommandLine(CommandLine);
  ModifiedCommandLine.emplace_back(InputPath);

  return std::make_pair(OverlayFS, ModifiedCommandLine);
}

static std::pair<IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem>,
                 std::vector<std::string>>
initVFSForByNameScanning(IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS,
                         ArrayRef<std::string> CommandLine,
                         StringRef WorkingDirectory, StringRef ModuleName) {
  // If we're scanning based on a module name alone, we don't expect the client
  // to provide us with an input file. However, the driver really wants to have
  // one. Let's just make it up to make the driver happy.
  auto OverlayFS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(BaseFS);
  // Reset what might have been modified in the previous worker invocation.
  OverlayFS->setCurrentWorkingDirectory(WorkingDirectory);
  auto InMemoryFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  InMemoryFS->setCurrentWorkingDirectory(WorkingDirectory);
  SmallString<128> FakeInputPath;
  // TODO: We should retry the creation if the path already exists.
  llvm::sys::fs::createUniquePath(ModuleName + "-%%%%%%%%.input", FakeInputPath,
                                  /*MakeAbsolute=*/false);
  InMemoryFS->addFile(FakeInputPath, 0, llvm::MemoryBuffer::getMemBuffer(""));
  OverlayFS->pushOverlay(InMemoryFS);

  std::vector<std::string> ModifiedCommandLine(CommandLine);
  ModifiedCommandLine.emplace_back(FakeInputPath);

  return std::make_pair(OverlayFS, ModifiedCommandLine);
}

static llvm::Error makeErrorFromDiagnosticsOS(
    TextDiagnosticsPrinterWithOutput &DiagPrinterWithOS) {
  return llvm::make_error<llvm::StringError>(
      DiagPrinterWithOS.DiagnosticsOS.str(), llvm::inconvertibleErrorCode());
}

static bool computeDependenciesForDriverCommandLine(
    DependencyScanningWorker &Worker, StringRef WorkingDirectory,
    ArrayRef<std::string> CommandLine, DependencyConsumer &Consumer,
    DependencyActionController &Controller, DiagnosticConsumer &DiagConsumer,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> ScanFS) {
  Worker.getVFS().setCurrentWorkingDirectory(WorkingDirectory);

  DignosticsEngineWithDiagOpts DiagEngineWithDiagOpts(
      CommandLine, &Worker.getVFS(), DiagConsumer);
  auto &Diags = *DiagEngineWithDiagOpts.DiagEngine;

  // Compilation holds a non-owning a reference to the Driver, hence we need to
  // keep the Driver alive when we use Compilation. Arguments to commands may be
  // owned by Alloc when expanded from response files.
  llvm::BumpPtrAllocator Alloc;
  const auto [Driver, Compilation] = buildCompilation(
      CommandLine, *DiagEngineWithDiagOpts.DiagEngine, &Worker.getVFS(), Alloc);
  if (!Compilation)
    return false;

  const auto CC1Commands = llvm::to_vector(
      llvm::map_range(Compilation->getJobs(), buildCC1CommandLine));

  return Worker.computeDependencies(WorkingDirectory, CC1Commands, Consumer,
                                    Controller, Diags, ScanFS);
}

static llvm::Error computeDependenciesOrError(
    DependencyScanningWorker &Worker, StringRef WorkingDirectory,
    ArrayRef<std::string> CommandLine, DependencyConsumer &Consumer,
    DependencyActionController &Controller,
    std::optional<llvm::MemoryBufferRef> TUBuffer = std::nullopt) {
  auto [OverlayFS, FinalCommandLine] = [&]() {
    if (TUBuffer)
      return initVFSForTUBufferScanning(&Worker.getVFS(), CommandLine,
                                        WorkingDirectory, *TUBuffer);
    return std::make_pair(
        IntrusiveRefCntPtr<llvm::vfs::FileSystem>(&Worker.getVFS()),
        std::vector<std::string>(CommandLine.begin(), CommandLine.end()));
  }();

  TextDiagnosticsPrinterWithOutput DiagPrinterWithOS(CommandLine);

  const auto IsCC1Input = (FinalCommandLine[1] == "-cc1");
  const auto Success =
      IsCC1Input
          ? Worker.computeDependencies(WorkingDirectory, FinalCommandLine,
                                       Consumer, Controller,
                                       DiagPrinterWithOS.DiagPrinter, OverlayFS)
          : computeDependenciesForDriverCommandLine(
                Worker, WorkingDirectory, FinalCommandLine, Consumer,
                Controller, DiagPrinterWithOS.DiagPrinter, OverlayFS);

  if (!Success)
    return makeErrorFromDiagnosticsOS(DiagPrinterWithOS);
  return llvm::Error::success();
}

llvm::Expected<std::string> DependencyScanningTool::getDependencyFile(
    const std::vector<std::string> &CommandLine, StringRef CWD) {
  MakeDependencyPrinterConsumer Consumer;
  CallbackActionController Controller(nullptr);
  if (auto Result = computeDependenciesOrError(Worker, CWD, CommandLine,
                                               Consumer, Controller))
    return std::move(Result);
  std::string Output;
  Consumer.printDependencies(Output);
  return Output;
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
  if (auto Result = computeDependenciesOrError(Worker, CWD, Command.CommandLine,
                                               Consumer, Controller))
    return std::move(Result);

  MakeformatOutputPath = Consumer.getMakeFormatDependencyOutputPath();
  if (!MakeformatOutputPath.empty())
    Consumer.printDependencies(MakeformatOutput);
  return Rule;
}

llvm::Expected<TranslationUnitDeps>
DependencyScanningTool::getTranslationUnitDependencies(
    const std::vector<std::string> &CommandLine, StringRef CWD,
    const llvm::DenseSet<ModuleID> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput,
    std::optional<llvm::MemoryBufferRef> TUBuffer) {
  FullDependencyConsumer Consumer(AlreadySeen);
  CallbackActionController Controller(LookupModuleOutput);
  if (auto Result = computeDependenciesOrError(Worker, CWD, CommandLine,
                                               Consumer, Controller, TUBuffer))
    return std::move(Result);
  return Consumer.takeTranslationUnitDeps();
}

llvm::Expected<TranslationUnitDeps>
DependencyScanningTool::getModuleDependencies(
    StringRef ModuleName, const std::vector<std::string> &CommandLine,
    StringRef CWD, const llvm::DenseSet<ModuleID> &AlreadySeen,
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

static std::optional<std::vector<std::string>> getFirstCC1CommandLine(
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
  const auto CC1CommandLineRange = llvm::map_range(
      llvm::make_filter_range(Compilation->getJobs(), IsClangCmd),
      buildCC1CommandLine);

  if (CC1CommandLineRange.empty())
    return std::nullopt;
  return *CC1CommandLineRange.begin();
}

llvm::Error
DependencyScanningTool::initializeCompilerInstanceWithContextOrError(
    StringRef CWD, ArrayRef<std::string> CommandLine) {
  // For by name scanning, we allow command lines without an actual input file
  // by adding an in-memory placeholder input.
  auto OverlayFSAndArgs = initVFSForByNameScanning(
      &Worker.getVFS(), CommandLine, CWD, "ScanningByName");
  auto &OverlayFS = OverlayFSAndArgs.first;
  const auto &ModifiedCommandLine = OverlayFSAndArgs.second;

  DiagPrinterWithOS =
      std::make_unique<TextDiagnosticsPrinterWithOutput>(CommandLine);
  auto DiagEngineWithCmdAndOpts =
      std::make_unique<DignosticsEngineWithDiagOpts>(
          CommandLine, OverlayFS, DiagPrinterWithOS->DiagPrinter);

  const auto InitWithCommandLine =
      [&](ArrayRef<std::string> CommandLine) -> llvm::Error {
    if (Worker.initializeCompilerInstanceWithContext(
            CWD, CommandLine, std::move(DiagEngineWithCmdAndOpts), OverlayFS))
      return llvm::Error::success();
    return makeErrorFromDiagnosticsOS(*DiagPrinterWithOS);
  };

  if (CommandLine.size() >= 2 && CommandLine[1] == "-cc1")
    return InitWithCommandLine(CommandLine);

  const auto MaybeFirstCC1 = getFirstCC1CommandLine(
      ModifiedCommandLine, *DiagEngineWithCmdAndOpts->DiagEngine, OverlayFS);
  if (!MaybeFirstCC1)
    return makeErrorFromDiagnosticsOS(*DiagPrinterWithOS);
  return InitWithCommandLine(*MaybeFirstCC1);
}

llvm::Expected<TranslationUnitDeps>
DependencyScanningTool::computeDependenciesByNameWithContextOrError(
    StringRef ModuleName, const llvm::DenseSet<ModuleID> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput) {
  FullDependencyConsumer Consumer(AlreadySeen);
  CallbackActionController Controller(LookupModuleOutput);
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
