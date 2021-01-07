//===- CompilerInvocation.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "clang/Basic/AllDiagnostics.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

using namespace Fortran::frontend;

//===----------------------------------------------------------------------===//
// Initialization.
//===----------------------------------------------------------------------===//
CompilerInvocationBase::CompilerInvocationBase()
    : diagnosticOpts_(new clang::DiagnosticOptions()),
      preprocessorOpts_(new PreprocessorOptions()) {}

CompilerInvocationBase::CompilerInvocationBase(const CompilerInvocationBase &x)
    : diagnosticOpts_(new clang::DiagnosticOptions(x.GetDiagnosticOpts())),
      preprocessorOpts_(new PreprocessorOptions(x.preprocessorOpts())) {}

CompilerInvocationBase::~CompilerInvocationBase() = default;

//===----------------------------------------------------------------------===//
// Deserialization (from args)
//===----------------------------------------------------------------------===//
static bool parseShowColorsArgs(
    const llvm::opt::ArgList &args, bool defaultColor) {
  // Color diagnostics default to auto ("on" if terminal supports) in the driver
  // but default to off in cc1, needing an explicit OPT_fdiagnostics_color.
  // Support both clang's -f[no-]color-diagnostics and gcc's
  // -f[no-]diagnostics-colors[=never|always|auto].
  enum {
    Colors_On,
    Colors_Off,
    Colors_Auto
  } ShowColors = defaultColor ? Colors_Auto : Colors_Off;

  for (auto *a : args) {
    const llvm::opt::Option &O = a->getOption();
    if (O.matches(clang::driver::options::OPT_fcolor_diagnostics) ||
        O.matches(clang::driver::options::OPT_fdiagnostics_color)) {
      ShowColors = Colors_On;
    } else if (O.matches(clang::driver::options::OPT_fno_color_diagnostics) ||
        O.matches(clang::driver::options::OPT_fno_diagnostics_color)) {
      ShowColors = Colors_Off;
    } else if (O.matches(clang::driver::options::OPT_fdiagnostics_color_EQ)) {
      llvm::StringRef value(a->getValue());
      if (value == "always")
        ShowColors = Colors_On;
      else if (value == "never")
        ShowColors = Colors_Off;
      else if (value == "auto")
        ShowColors = Colors_Auto;
    }
  }

  return ShowColors == Colors_On ||
      (ShowColors == Colors_Auto && llvm::sys::Process::StandardErrHasColors());
}

bool Fortran::frontend::ParseDiagnosticArgs(clang::DiagnosticOptions &opts,
    llvm::opt::ArgList &args, bool defaultDiagColor) {
  opts.ShowColors = parseShowColorsArgs(args, defaultDiagColor);

  return true;
}

static InputKind ParseFrontendArgs(FrontendOptions &opts,
    llvm::opt::ArgList &args, clang::DiagnosticsEngine &diags) {
  // Identify the action (i.e. opts.ProgramAction)
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_Action_Group)) {
    switch (a->getOption().getID()) {
    default: {
      llvm_unreachable("Invalid option in group!");
    }
    case clang::driver::options::OPT_test_io:
      opts.programAction_ = InputOutputTest;
      break;
    case clang::driver::options::OPT_E:
      opts.programAction_ = PrintPreprocessedInput;
      break;
    case clang::driver::options::OPT_fsyntax_only:
      opts.programAction_ = ParseSyntaxOnly;
      break;
    case clang::driver::options::OPT_emit_obj:
      opts.programAction_ = EmitObj;
      break;

      // TODO:
      // case calng::driver::options::OPT_emit_llvm:
      // case clang::driver::options::OPT_emit_llvm_only:
      // case clang::driver::options::OPT_emit_codegen_only:
      // case clang::driver::options::OPT_emit_module:
      // (...)
    }
  }

  opts.outputFile_ = args.getLastArgValue(clang::driver::options::OPT_o);
  opts.showHelp_ = args.hasArg(clang::driver::options::OPT_help);
  opts.showVersion_ = args.hasArg(clang::driver::options::OPT_version);

  // Get the input kind (from the value passed via `-x`)
  InputKind dashX(Language::Unknown);
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_x)) {
    llvm::StringRef XValue = a->getValue();
    // Principal languages.
    dashX = llvm::StringSwitch<InputKind>(XValue)
                .Case("f90", Language::Fortran)
                .Default(Language::Unknown);

    // Some special cases cannot be combined with suffixes.
    if (dashX.IsUnknown())
      dashX = llvm::StringSwitch<InputKind>(XValue)
                  .Case("ir", Language::LLVM_IR)
                  .Default(Language::Unknown);

    if (dashX.IsUnknown())
      diags.Report(clang::diag::err_drv_invalid_value)
          << a->getAsString(args) << a->getValue();
  }

  // Collect the input files and save them in our instance of FrontendOptions.
  std::vector<std::string> inputs =
      args.getAllArgValues(clang::driver::options::OPT_INPUT);
  opts.inputs_.clear();
  if (inputs.empty())
    // '-' is the default input if none is given.
    inputs.push_back("-");
  for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
    InputKind ik = dashX;
    if (ik.IsUnknown()) {
      ik = FrontendOptions::GetInputKindForExtension(
          llvm::StringRef(inputs[i]).rsplit('.').second);
      if (ik.IsUnknown())
        ik = Language::Unknown;
      if (i == 0)
        dashX = ik;
    }

    opts.inputs_.emplace_back(std::move(inputs[i]), ik);
  }
  return dashX;
}

/// Parses all preprocessor input arguments and populates the preprocessor
/// options accordingly.
///
/// \param [in] opts The preprocessor options instance
/// \param [out] args The list of input arguments
static void parsePreprocessorArgs(
    Fortran::frontend::PreprocessorOptions &opts, llvm::opt::ArgList &args) {
  // Add macros from the command line.
  for (const auto *currentArg : args.filtered(
           clang::driver::options::OPT_D, clang::driver::options::OPT_U)) {
    if (currentArg->getOption().matches(clang::driver::options::OPT_D)) {
      opts.addMacroDef(currentArg->getValue());
    } else {
      opts.addMacroUndef(currentArg->getValue());
    }
  }
}

bool CompilerInvocation::CreateFromArgs(CompilerInvocation &res,
    llvm::ArrayRef<const char *> commandLineArgs,
    clang::DiagnosticsEngine &diags) {

  bool success = true;

  // Parse the arguments
  const llvm::opt::OptTable &opts = clang::driver::getDriverOptTable();
  const unsigned includedFlagsBitmask =
      clang::driver::options::FC1Option;
  unsigned missingArgIndex, missingArgCount;
  llvm::opt::InputArgList args = opts.ParseArgs(
      commandLineArgs, missingArgIndex, missingArgCount, includedFlagsBitmask);

  // Issue errors on unknown arguments
  for (const auto *a : args.filtered(clang::driver::options::OPT_UNKNOWN)) {
    auto argString = a->getAsString(args);
    std::string nearest;
    if (opts.findNearest(argString, nearest, includedFlagsBitmask) > 1)
      diags.Report(clang::diag::err_drv_unknown_argument) << argString;
    else
      diags.Report(clang::diag::err_drv_unknown_argument_with_suggestion)
          << argString << nearest;
    success = false;
  }

  // Parse the frontend args
  ParseFrontendArgs(res.frontendOpts(), args, diags);
  // Parse the preprocessor args
  parsePreprocessorArgs(res.preprocessorOpts(), args);

  return success;
}

/// Collect the macro definitions provided by the given preprocessor
/// options into the parser options.
///
/// \param [in] ppOpts The preprocessor options
/// \param [out] opts The fortran options
static void collectMacroDefinitions(
    const PreprocessorOptions &ppOpts, Fortran::parser::Options &opts) {
  for (unsigned i = 0, n = ppOpts.macros.size(); i != n; ++i) {
    llvm::StringRef macro = ppOpts.macros[i].first;
    bool isUndef = ppOpts.macros[i].second;

    std::pair<llvm::StringRef, llvm::StringRef> macroPair = macro.split('=');
    llvm::StringRef macroName = macroPair.first;
    llvm::StringRef macroBody = macroPair.second;

    // For an #undef'd macro, we only care about the name.
    if (isUndef) {
      opts.predefinitions.emplace_back(
          macroName.str(), std::optional<std::string>{});
      continue;
    }

    // For a #define'd macro, figure out the actual definition.
    if (macroName.size() == macro.size())
      macroBody = "1";
    else {
      // Note: GCC drops anything following an end-of-line character.
      llvm::StringRef::size_type End = macroBody.find_first_of("\n\r");
      macroBody = macroBody.substr(0, End);
    }
    opts.predefinitions.emplace_back(
        macroName, std::optional<std::string>(macroBody.str()));
  }
}

void CompilerInvocation::SetDefaultFortranOpts() {
  auto &fortranOptions = fortranOpts();

  // These defaults are based on the defaults in f18/f18.cpp.
  std::vector<std::string> searchDirectories{"."s};
  fortranOptions.searchDirectories = searchDirectories;
  fortranOptions.isFixedForm = false;
}

void CompilerInvocation::setFortranOpts() {
  auto &fortranOptions = fortranOpts();
  const auto &preprocessorOptions = preprocessorOpts();

  collectMacroDefinitions(preprocessorOptions, fortranOptions);
}
