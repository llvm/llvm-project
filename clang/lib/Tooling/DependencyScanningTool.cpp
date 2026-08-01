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
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
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
      DependencyScanningWorker::MaxNumOfByNameQueries, ' ');

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
  CallbackActionController Controller(LookupModuleOutput);

  // If we are scanning from a TUBuffer, create an overlay filesystem with the
  // input as an in-memory file and add it to the command line.
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> OverlayFS = nullptr;
  std::vector<std::string> CommandLineWithTUBufferInput;
  if (TUBuffer) {
    std::tie(OverlayFS, CommandLineWithTUBufferInput) =
        initVFSForTUBufferScanning(CommandLine, *TUBuffer);
    CommandLine = CommandLineWithTUBufferInput;
  }

  if (!computeDependencies(Worker, CWD, CommandLine, Consumer, Controller,
                           DiagConsumer, std::move(OverlayFS)))
    return std::nullopt;
  return Consumer.takeTranslationUnitDeps();
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

bool DependencyScanningTool::getByNameDependencies(
    StringRef CWD, ArrayRef<std::string> CommandLine,
    DiagnosticConsumer &DiagConsumer, DependencyActionController &Controller,
    llvm::function_ref<std::optional<std::string>()> getNextName,
    DependencyConsumer &DepConsumer) {
  auto [OverlayFS, ModifiedCommandLine] = initVFSForByNameScanning(CommandLine);
  auto FS = Worker.makeEffectiveVFS(CWD, OverlayFS);
  std::vector<std::string> CC1CommandLine;
  if (ModifiedCommandLine.size() >= 2 && ModifiedCommandLine[1] == "-cc1") {
    CC1CommandLine = std::move(ModifiedCommandLine);
  } else {
    // Driver-style (or ill-formed): lower to a cc1 command line, or diagnose.
    DiagnosticsEngineWithDiagOpts DiagEngineWithOpts(ModifiedCommandLine, FS,
                                                     DiagConsumer);
    auto MaybeFirstCC1 = getFirstCC1CommandLine(
        ModifiedCommandLine, *DiagEngineWithOpts.DiagEngine, FS);
    if (!MaybeFirstCC1)
      return false;
    CC1CommandLine.assign(MaybeFirstCC1->begin(), MaybeFirstCC1->end());
  }

  return Worker.computeDependenciesByName(CWD, CC1CommandLine,
                                          std::move(OverlayFS), DiagConsumer,
                                          Controller, getNextName, DepConsumer);
}
