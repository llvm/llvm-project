//===- CompilerInvocation.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Common/Fortran-features.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Semantics/semantics.h"
#include "flang/Version.inc"
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
#include <memory>

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

  // By default the frontend driver creates a ParseSyntaxOnly action.
  opts.programAction_ = ParseSyntaxOnly;

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
    case clang::driver::options::OPT_fdebug_unparse:
      opts.programAction_ = DebugUnparse;
      break;
    case clang::driver::options::OPT_fdebug_unparse_with_symbols:
      opts.programAction_ = DebugUnparseWithSymbols;
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

  // Set fortranForm_ based on options -ffree-form and -ffixed-form.
  if (const auto *arg = args.getLastArg(clang::driver::options::OPT_ffixed_form,
          clang::driver::options::OPT_ffree_form)) {
    opts.fortranForm_ =
        arg->getOption().matches(clang::driver::options::OPT_ffixed_form)
        ? FortranForm::FixedForm
        : FortranForm::FreeForm;
  }

  // Set fixedFormColumns_ based on -ffixed-line-length=<value>
  if (const auto *arg =
          args.getLastArg(clang::driver::options::OPT_ffixed_line_length_EQ)) {
    llvm::StringRef argValue = llvm::StringRef(arg->getValue());
    std::int64_t columns = -1;
    if (argValue == "none") {
      columns = 0;
    } else if (argValue.getAsInteger(/*Radix=*/10, columns)) {
      columns = -1;
    }
    if (columns < 0) {
      diags.Report(clang::diag::err_drv_invalid_value_with_suggestion)
          << arg->getOption().getName() << arg->getValue()
          << "value must be 'none' or a non-negative integer";
    } else if (columns == 0) {
      opts.fixedFormColumns_ = 1000000;
    } else if (columns < 7) {
      diags.Report(clang::diag::err_drv_invalid_value_with_suggestion)
          << arg->getOption().getName() << arg->getValue()
          << "value must be at least seven";
    } else {
      opts.fixedFormColumns_ = columns;
    }
  }

  // Extensions
  if (args.hasArg(clang::driver::options::OPT_fopenacc)) {
    opts.features_.Enable(Fortran::common::LanguageFeature::OpenACC);
  }
  if (args.hasArg(clang::driver::options::OPT_fopenmp)) {
    opts.features_.Enable(Fortran::common::LanguageFeature::OpenMP);
  }
  if (const llvm::opt::Arg *arg =
          args.getLastArg(clang::driver::options::OPT_fimplicit_none,
              clang::driver::options::OPT_fno_implicit_none)) {
    opts.features_.Enable(
        Fortran::common::LanguageFeature::ImplicitNoneTypeAlways,
        arg->getOption().matches(clang::driver::options::OPT_fimplicit_none));
  }
  if (const llvm::opt::Arg *arg =
          args.getLastArg(clang::driver::options::OPT_fbackslash,
              clang::driver::options::OPT_fno_backslash)) {
    opts.features_.Enable(Fortran::common::LanguageFeature::BackslashEscapes,
        arg->getOption().matches(clang::driver::options::OPT_fbackslash));
  }
  if (const llvm::opt::Arg *arg =
          args.getLastArg(clang::driver::options::OPT_flogical_abbreviations,
              clang::driver::options::OPT_fno_logical_abbreviations)) {
    opts.features_.Enable(
        Fortran::common::LanguageFeature::LogicalAbbreviations,
        arg->getOption().matches(
            clang::driver::options::OPT_flogical_abbreviations));
  }
  if (const llvm::opt::Arg *arg =
          args.getLastArg(clang::driver::options::OPT_fxor_operator,
              clang::driver::options::OPT_fno_xor_operator)) {
    opts.features_.Enable(Fortran::common::LanguageFeature::XOROperator,
        arg->getOption().matches(clang::driver::options::OPT_fxor_operator));
  }
  if (args.hasArg(
          clang::driver::options::OPT_falternative_parameter_statement)) {
    opts.features_.Enable(Fortran::common::LanguageFeature::OldStyleParameter);
  }
  if (const llvm::opt::Arg *arg =
          args.getLastArg(clang::driver::options::OPT_finput_charset_EQ)) {
    llvm::StringRef argValue = arg->getValue();
    if (argValue == "utf-8") {
      opts.encoding_ = Fortran::parser::Encoding::UTF_8;
    } else if (argValue == "latin-1") {
      opts.encoding_ = Fortran::parser::Encoding::LATIN_1;
    } else {
      diags.Report(clang::diag::err_drv_invalid_value)
          << arg->getAsString(args) << argValue;
    }
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

  // Add the ordered list of -I's.
  for (const auto *currentArg : args.filtered(clang::driver::options::OPT_I))
    opts.searchDirectoriesFromDashI.emplace_back(currentArg->getValue());
}

/// Parses all semantic related arguments and populates the variables
/// options accordingly.
static void parseSemaArgs(std::string &moduleDir, llvm::opt::ArgList &args,
    clang::DiagnosticsEngine &diags) {

  auto moduleDirList =
      args.getAllArgValues(clang::driver::options::OPT_module_dir);
  // User can only specify -J/-module-dir once
  // https://gcc.gnu.org/onlinedocs/gfortran/Directory-Options.html
  if (moduleDirList.size() > 1) {
    const unsigned diagID =
        diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
            "Only one '-module-dir/-J' option allowed");
    diags.Report(diagID);
  }
  if (moduleDirList.size() == 1)
    moduleDir = moduleDirList[0];
}

bool CompilerInvocation::CreateFromArgs(CompilerInvocation &res,
    llvm::ArrayRef<const char *> commandLineArgs,
    clang::DiagnosticsEngine &diags) {

  bool success = true;

  // Parse the arguments
  const llvm::opt::OptTable &opts = clang::driver::getDriverOptTable();
  const unsigned includedFlagsBitmask = clang::driver::options::FC1Option;
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
  // Parse semantic args
  parseSemaArgs(res.moduleDir(), args, diags);

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

// TODO: When expanding this method, consider creating a dedicated API for
// this. Also at some point we will need to differentiate between different
// targets and add dedicated predefines for each.
void CompilerInvocation::setDefaultPredefinitions() {
  auto &fortranOptions = fortranOpts();
  const auto &frontendOptions = frontendOpts();

  // Populate the macro list with version numbers and other predefinitions.
  fortranOptions.predefinitions.emplace_back("__flang__", "1");
  fortranOptions.predefinitions.emplace_back(
      "__flang_major__", FLANG_VERSION_MAJOR_STRING);
  fortranOptions.predefinitions.emplace_back(
      "__flang_minor__", FLANG_VERSION_MINOR_STRING);
  fortranOptions.predefinitions.emplace_back(
      "__flang_patchlevel__", FLANG_VERSION_PATCHLEVEL_STRING);

  // Add predefinitions based on extensions enabled
  if (frontendOptions.features_.IsEnabled(
          Fortran::common::LanguageFeature::OpenACC)) {
    fortranOptions.predefinitions.emplace_back("_OPENACC", "202011");
  }
  if (frontendOptions.features_.IsEnabled(
          Fortran::common::LanguageFeature::OpenMP)) {
    fortranOptions.predefinitions.emplace_back("_OPENMP", "201511");
  }
}

void CompilerInvocation::setFortranOpts() {
  auto &fortranOptions = fortranOpts();
  const auto &frontendOptions = frontendOpts();
  const auto &preprocessorOptions = preprocessorOpts();
  auto &moduleDirJ = moduleDir();

  if (frontendOptions.fortranForm_ != FortranForm::Unknown) {
    fortranOptions.isFixedForm =
        frontendOptions.fortranForm_ == FortranForm::FixedForm;
  }
  fortranOptions.fixedFormColumns = frontendOptions.fixedFormColumns_;

  fortranOptions.features = frontendOptions.features_;
  fortranOptions.encoding = frontendOptions.encoding_;

  collectMacroDefinitions(preprocessorOptions, fortranOptions);

  fortranOptions.searchDirectories.insert(
      fortranOptions.searchDirectories.end(),
      preprocessorOptions.searchDirectoriesFromDashI.begin(),
      preprocessorOptions.searchDirectoriesFromDashI.end());

  // Add the directory supplied through -J/-module-dir to the list of search
  // directories
  if (moduleDirJ.compare(".") != 0)
    fortranOptions.searchDirectories.emplace_back(moduleDirJ);
}

void CompilerInvocation::setSemanticsOpts(
    Fortran::parser::AllCookedSources &allCookedSources) {
  const auto &fortranOptions = fortranOpts();

  semanticsContext_ = std::make_unique<semantics::SemanticsContext>(
      *(new Fortran::common::IntrinsicTypeDefaultKinds()),
      fortranOptions.features, allCookedSources);

  auto &moduleDirJ = moduleDir();
  semanticsContext_->set_moduleDirectory(moduleDirJ)
      .set_searchDirectories(fortranOptions.searchDirectories);
}
