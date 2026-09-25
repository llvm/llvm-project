//===--- FlangTidyMain.cpp - flang-tidy -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../FlangTidy.h"
#include "../FlangTidyForceLinker.h"
#include "../FlangTidyOptions.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <sstream>
#include <vector>

// Frontend driver
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/TextDiagnosticBuffer.h"

namespace Fortran::tidy {

static llvm::cl::list<std::string>
    SourcePaths(llvm::cl::Positional,
                llvm::cl::desc("<source0> [... <sourceN>]"),
                llvm::cl::OneOrMore, llvm::cl::value_desc("source files"),
                llvm::cl::sub(llvm::cl::SubCommand::getAll()));

static llvm::cl::opt<std::string>
    CheckOption("checks",
                llvm::cl::desc("Comma-separated list of checks to enable. "
                               "Overrides configuration file settings."),
                llvm::cl::init(""), llvm::cl::value_desc("check list"));

static llvm::cl::opt<std::string> ConfigOption(
    "config",
    llvm::cl::desc(
        "Specify configuration in YAML format: "
        "-config=\"{Checks: '*', CognitiveComplexityThreshold: 30}\" "
        "When empty, flang-tidy will look for .flang-tidy files."),
    llvm::cl::init(""), llvm::cl::value_desc("yaml config"));

static llvm::cl::opt<std::string> ConfigFile(
    "config-file",
    llvm::cl::desc("Specify the path of .flang-tidy or custom config file"),
    llvm::cl::init(""), llvm::cl::value_desc("filename"));

static llvm::cl::opt<bool>
    DumpConfig("dump-config",
               llvm::cl::desc("Dump configuration in YAML format to stdout"),
               llvm::cl::init(false));

static llvm::cl::list<std::string> ArgsBefore(
    "extra-arg-before",
    llvm::cl::desc(
        "Additional argument to prepend to the compiler command line"),
    llvm::cl::ZeroOrMore, llvm::cl::sub(llvm::cl::SubCommand::getAll()));

static llvm::cl::list<std::string>
    ArgsAfter("extra-arg",
              llvm::cl::desc(
                  "Additional argument to append to the compiler command line"),
              llvm::cl::ZeroOrMore,
              llvm::cl::sub(llvm::cl::SubCommand::getAll()));

static llvm::cl::opt<std::string>
    WarningsAsErrors("warnings-as-errors",
                     llvm::cl::desc("Comma-separated list of checks for which "
                                    "to turn warnings into errors"),
                     llvm::cl::init(""));

static std::string GetFlangToolCommand() {
  static int Dummy;
  std::string FlangExecutable =
      llvm::sys::fs::getMainExecutable("flang", (void *)&Dummy);
  llvm::SmallString<128> FlangToolPath;
  FlangToolPath = llvm::sys::path::parent_path(FlangExecutable);
  llvm::sys::path::append(FlangToolPath, "flang-tool");
  return std::string(FlangToolPath);
}

static bool stripPositionalArgs(std::vector<const char *> Args,
                                std::vector<std::string> &Result,
                                std::string &ErrorMsg) {
  auto flang = std::make_unique<Fortran::frontend::CompilerInstance>();

  // Create diagnostics engine
  flang->createDiagnostics();
  if (!flang->hasDiagnostics()) {
    llvm::errs() << "Failed to create diagnostics engine\n";
    return false;
  }

  // Capture diagnostics
  frontend::TextDiagnosticBuffer *diagsBuffer =
      new frontend::TextDiagnosticBuffer;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(
      new clang::DiagnosticIDs());
  clang::DiagnosticOptions diagOpts;
  clang::DiagnosticsEngine diags(diagID, diagOpts, diagsBuffer);

  // Insert Flang tool command
  std::string Argv0 = GetFlangToolCommand();
  Args.insert(Args.begin(), Argv0.c_str());

  // Add a dummy file to ensure at least one compilation job
  Args.push_back("placeholder.f90");

  // Remove -c flags if present
  Args.erase(std::remove_if(
                 Args.begin(), Args.end(),
                 [](const char *arg) { return llvm::StringRef(arg) == "-c"; }),
             Args.end());

  // Create compiler invocation
  bool success = Fortran::frontend::CompilerInvocation::createFromArgs(
      flang->getInvocation(), Args, diags, Argv0.c_str());

  if (!success) {
    ErrorMsg = "Failed to create compiler invocation\n";
    // Flush diagnostic
    diagsBuffer->flushDiagnostics(flang->getDiagnostics());
    return false;
  }

  // Get the list of input files from Flang's frontend options
  std::vector<std::string> inputs;
  for (const auto &input : flang->getFrontendOpts().inputs) {
    inputs.push_back(input.getFile().str());
  }

  if (inputs.empty()) {
    ErrorMsg = "warning: no compile jobs found\n";
    return false;
  }

  // Remove input files from Args
  std::vector<const char *>::iterator End = llvm::remove_if(
      Args, [&](llvm::StringRef S) { return llvm::is_contained(inputs, S); });

  // Store the filtered arguments
  Result = std::vector<std::string>(Args.begin(), End);
  return true;
}

static std::vector<std::string>
loadFromCommandLine(int &Argc, const char *const *Argv, std::string &ErrorMsg) {
  ErrorMsg.clear();
  if (Argc == 0)
    return {};
  const char *const *DoubleDash =
      std::find(Argv, Argv + Argc, llvm::StringRef("--"));
  if (DoubleDash == Argv + Argc)
    return {};
  std::vector<const char *> CommandLine(DoubleDash + 1, Argv + Argc);
  Argc = DoubleDash - Argv;

  std::vector<std::string> StrippedArgs;
  if (!stripPositionalArgs(CommandLine, StrippedArgs, ErrorMsg))
    return {};
  return StrippedArgs;
}

static std::unique_ptr<FlangTidyOptionsProvider>
createOptionsProvider(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS) {
  FlangTidyGlobalOptions GlobalOptions;

  FlangTidyOptions DefaultOptions = FlangTidyOptions::getDefaults();

  FlangTidyOptions OverrideOptions;
  if (!CheckOption.empty())
    OverrideOptions.Checks =
        CheckOption; // This completely overrides, not merges

  if (!WarningsAsErrors.empty())
    OverrideOptions.WarningsAsErrors =
        WarningsAsErrors; // This completely overrides, not merges

  auto LoadConfig =
      [&](llvm::StringRef Configuration,
          llvm::StringRef Source) -> std::unique_ptr<FlangTidyOptionsProvider> {
    llvm::ErrorOr<FlangTidyOptions> ParsedConfig =
        parseConfiguration(llvm::MemoryBufferRef(Configuration, Source));
    if (ParsedConfig)
      return std::make_unique<ConfigOptionsProvider>(
          std::move(GlobalOptions),
          FlangTidyOptions::getDefaults().merge(DefaultOptions, 0),
          std::move(*ParsedConfig), std::move(OverrideOptions), std::move(FS));
    llvm::errs() << "Error: invalid configuration specified.\n"
                 << ParsedConfig.getError().message() << "\n";
    return nullptr;
  };

  if (!ConfigFile.empty()) {
    if (!ConfigOption.empty()) {
      llvm::errs() << "Error: --config-file and --config are "
                      "mutually exclusive. Specify only one.\n";
      return nullptr;
    }

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Text =
        llvm::MemoryBuffer::getFile(ConfigFile);
    if (std::error_code EC = Text.getError()) {
      llvm::errs() << "Error: can't read config-file '" << ConfigFile
                   << "': " << EC.message() << "\n";
      return nullptr;
    }

    return LoadConfig((*Text)->getBuffer(), ConfigFile);
  }

  if (!ConfigOption.empty())
    return LoadConfig(ConfigOption, "<command-line-config>");

  return std::make_unique<FileOptionsProvider>(
      std::move(GlobalOptions), std::move(DefaultOptions),
      std::move(OverrideOptions), std::move(FS));
}

extern int flangTidyMain(int &argc, const char **argv) {
  llvm::InitLLVM X(argc, argv);

  std::string ErrorMessage;
  auto Compilations = loadFromCommandLine(argc, argv, ErrorMessage);

  if (!ErrorMessage.empty()) {
    llvm::outs() << ErrorMessage << "\n";
    return 1;
  }

  llvm::cl::ParseCommandLineOptions(
      argc, argv, "flang-tidy: A Fortran source analysis tool\n");

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS =
      llvm::vfs::getRealFileSystem();

  auto OwningOptionsProvider = createOptionsProvider(BaseFS);
  auto *OptionsProvider = OwningOptionsProvider.get();
  if (!OptionsProvider)
    return 1;

  llvm::StringRef FileName = "dummy.f90";
  if (!SourcePaths.empty())
    FileName = SourcePaths[0];

  FlangTidyOptions EffectiveOptions = OptionsProvider->getOptions(FileName);

  if (DumpConfig) {
    llvm::outs() << configurationAsText(EffectiveOptions) << "\n";
    return 0;
  }

  EffectiveOptions.sourcePaths.assign(SourcePaths.begin(), SourcePaths.end());
  EffectiveOptions.argv0 = argv[0];

  EffectiveOptions.parseChecksString();
  EffectiveOptions.parseWarningsAsErrorsString();

  for (const auto &sourcePath : EffectiveOptions.sourcePaths) {
    if (!llvm::sys::fs::exists(sourcePath)) {
      llvm::errs() << "Error: File not found: " << sourcePath << "\n";
      return 1;
    }
  }

  // Add extra args from command line (these override config file settings)
  for (const auto &arg : ArgsBefore) {
    std::istringstream stream(arg);
    std::string subArg;
    while (stream >> subArg) {
      if (!EffectiveOptions.ExtraArgsBefore)
        EffectiveOptions.ExtraArgsBefore = std::vector<std::string>();
      EffectiveOptions.ExtraArgsBefore->push_back(subArg);
    }
  }

  for (const auto &arg : ArgsAfter) {
    std::istringstream stream(arg);
    std::string subArg;
    while (stream >> subArg) {
      if (!EffectiveOptions.ExtraArgs)
        EffectiveOptions.ExtraArgs = std::vector<std::string>();
      EffectiveOptions.ExtraArgs->push_back(subArg);
    }
  }

  // Add compilation args if present
  if (!Compilations.empty()) {
    assert(EffectiveOptions.sourcePaths.size() == 1);
    if (!EffectiveOptions.ExtraArgs)
      EffectiveOptions.ExtraArgs = std::vector<std::string>();
    EffectiveOptions.ExtraArgs->insert(EffectiveOptions.ExtraArgs->end(),
                                       Compilations.begin(),
                                       Compilations.end());
  }

  // Remove anything starting with --driver-mode
  if (EffectiveOptions.ExtraArgs) {
    EffectiveOptions.ExtraArgs->erase(
        std::remove_if(EffectiveOptions.ExtraArgs->begin(),
                       EffectiveOptions.ExtraArgs->end(),
                       [](std::string const &arg) {
                         return llvm::StringRef(arg).starts_with(
                             "--driver-mode");
                       }),
        EffectiveOptions.ExtraArgs->end());
  }

  // also remove --driver-mode from ExtraArgsBefore
  if (EffectiveOptions.ExtraArgsBefore) {
    EffectiveOptions.ExtraArgsBefore->erase(
        std::remove_if(EffectiveOptions.ExtraArgsBefore->begin(),
                       EffectiveOptions.ExtraArgsBefore->end(),
                       [](std::string const &arg) {
                         return llvm::StringRef(arg).starts_with(
                             "--driver-mode");
                       }),
        EffectiveOptions.ExtraArgsBefore->end());
  }

  return runFlangTidy(EffectiveOptions);
}

} // namespace Fortran::tidy
