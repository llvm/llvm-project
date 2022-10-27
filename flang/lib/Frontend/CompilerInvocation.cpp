//===- CompilerInvocation.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInvocation.h"
#include "flang/Common/Fortran-features.h"
#include "flang/Frontend/CodeGenOptions.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Frontend/TargetOptions.h"
#include "flang/Semantics/semantics.h"
#include "flang/Version.inc"
#include "clang/Basic/AllDiagnostics.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/OptionUtils.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace Fortran::frontend;

//===----------------------------------------------------------------------===//
// Initialization.
//===----------------------------------------------------------------------===//
CompilerInvocationBase::CompilerInvocationBase()
    : diagnosticOpts(new clang::DiagnosticOptions()),
      preprocessorOpts(new PreprocessorOptions()) {}

CompilerInvocationBase::CompilerInvocationBase(const CompilerInvocationBase &x)
    : diagnosticOpts(new clang::DiagnosticOptions(x.getDiagnosticOpts())),
      preprocessorOpts(new PreprocessorOptions(x.getPreprocessorOpts())) {}

CompilerInvocationBase::~CompilerInvocationBase() = default;

//===----------------------------------------------------------------------===//
// Deserialization (from args)
//===----------------------------------------------------------------------===//
static bool parseShowColorsArgs(const llvm::opt::ArgList &args,
                                bool defaultColor = true) {
  // Color diagnostics default to auto ("on" if terminal supports) in the
  // compiler driver `flang-new` but default to off in the frontend driver
  // `flang-new -fc1`, needing an explicit OPT_fdiagnostics_color.
  // Support both clang's -f[no-]color-diagnostics and gcc's
  // -f[no-]diagnostics-colors[=never|always|auto].
  enum {
    Colors_On,
    Colors_Off,
    Colors_Auto
  } showColors = defaultColor ? Colors_Auto : Colors_Off;

  for (auto *a : args) {
    const llvm::opt::Option &opt = a->getOption();
    if (opt.matches(clang::driver::options::OPT_fcolor_diagnostics)) {
      showColors = Colors_On;
    } else if (opt.matches(clang::driver::options::OPT_fno_color_diagnostics)) {
      showColors = Colors_Off;
    } else if (opt.matches(clang::driver::options::OPT_fdiagnostics_color_EQ)) {
      llvm::StringRef value(a->getValue());
      if (value == "always")
        showColors = Colors_On;
      else if (value == "never")
        showColors = Colors_Off;
      else if (value == "auto")
        showColors = Colors_Auto;
    }
  }

  return showColors == Colors_On ||
         (showColors == Colors_Auto &&
          llvm::sys::Process::StandardErrHasColors());
}

/// Extracts the optimisation level from \a args.
static unsigned getOptimizationLevel(llvm::opt::ArgList &args,
                                     clang::DiagnosticsEngine &diags) {
  unsigned defaultOpt = llvm::CodeGenOpt::None;

  if (llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_O_Group)) {
    if (a->getOption().matches(clang::driver::options::OPT_O0))
      return llvm::CodeGenOpt::None;

    assert(a->getOption().matches(clang::driver::options::OPT_O));

    return getLastArgIntValue(args, clang::driver::options::OPT_O, defaultOpt,
                              diags);
  }

  return defaultOpt;
}

bool Fortran::frontend::parseDiagnosticArgs(clang::DiagnosticOptions &opts,
                                            llvm::opt::ArgList &args) {
  opts.ShowColors = parseShowColorsArgs(args);

  return true;
}

static void parseCodeGenArgs(Fortran::frontend::CodeGenOptions &opts,
                             llvm::opt::ArgList &args,
                             clang::DiagnosticsEngine &diags) {
  opts.OptimizationLevel = getOptimizationLevel(args, diags);

  if (args.hasFlag(clang::driver::options::OPT_fdebug_pass_manager,
                   clang::driver::options::OPT_fno_debug_pass_manager, false))
    opts.DebugPassManager = 1;

  // -mrelocation-model option.
  if (const llvm::opt::Arg *A =
          args.getLastArg(clang::driver::options::OPT_mrelocation_model)) {
    llvm::StringRef ModelName = A->getValue();
    auto RM = llvm::StringSwitch<llvm::Optional<llvm::Reloc::Model>>(ModelName)
                  .Case("static", llvm::Reloc::Static)
                  .Case("pic", llvm::Reloc::PIC_)
                  .Case("dynamic-no-pic", llvm::Reloc::DynamicNoPIC)
                  .Case("ropi", llvm::Reloc::ROPI)
                  .Case("rwpi", llvm::Reloc::RWPI)
                  .Case("ropi-rwpi", llvm::Reloc::ROPI_RWPI)
                  .Default(llvm::None);
    if (RM.has_value())
      opts.setRelocationModel(*RM);
    else
      diags.Report(clang::diag::err_drv_invalid_value)
          << A->getAsString(args) << ModelName;
  }

  // -pic-level and -pic-is-pie option.
  if (int PICLevel = getLastArgIntValue(
          args, clang::driver::options::OPT_pic_level, 0, diags)) {
    if (PICLevel > 2)
      diags.Report(clang::diag::err_drv_invalid_value)
          << args.getLastArg(clang::driver::options::OPT_pic_level)
                 ->getAsString(args)
          << PICLevel;

    opts.PICLevel = PICLevel;
    if (args.hasArg(clang::driver::options::OPT_pic_is_pie))
      opts.IsPIE = 1;
  }
}

/// Parses all target input arguments and populates the target
/// options accordingly.
///
/// \param [in] opts The target options instance to update
/// \param [in] args The list of input arguments (from the compiler invocation)
static void parseTargetArgs(TargetOptions &opts, llvm::opt::ArgList &args) {
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_triple))
    opts.triple = a->getValue();
}

// Tweak the frontend configuration based on the frontend action
static void setUpFrontendBasedOnAction(FrontendOptions &opts) {
  if (opts.programAction == DebugDumpParsingLog)
    opts.instrumentedParse = true;

  if (opts.programAction == DebugDumpProvenance ||
      opts.programAction == Fortran::frontend::GetDefinition)
    opts.needProvenanceRangeToCharBlockMappings = true;
}

/// Parse the argument specified for the -fconvert=<value> option
static std::optional<const char *> parseConvertArg(const char *s) {
  return llvm::StringSwitch<std::optional<const char *>>(s)
      .Case("unknown", "UNKNOWN")
      .Case("native", "NATIVE")
      .Case("little-endian", "LITTLE_ENDIAN")
      .Case("big-endian", "BIG_ENDIAN")
      .Case("swap", "SWAP")
      .Default(std::nullopt);
}

static bool parseFrontendArgs(FrontendOptions &opts, llvm::opt::ArgList &args,
                              clang::DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // By default the frontend driver creates a ParseSyntaxOnly action.
  opts.programAction = ParseSyntaxOnly;

  // Treat multiple action options as an invocation error. Note that `clang
  // -cc1` does accept multiple action options, but will only consider the
  // rightmost one.
  if (args.hasMultipleArgs(clang::driver::options::OPT_Action_Group)) {
    const unsigned diagID = diags.getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Only one action option is allowed");
    diags.Report(diagID);
    return false;
  }

  // Identify the action (i.e. opts.ProgramAction)
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_Action_Group)) {
    switch (a->getOption().getID()) {
    default: {
      llvm_unreachable("Invalid option in group!");
    }
    case clang::driver::options::OPT_test_io:
      opts.programAction = InputOutputTest;
      break;
    case clang::driver::options::OPT_E:
      opts.programAction = PrintPreprocessedInput;
      break;
    case clang::driver::options::OPT_fsyntax_only:
      opts.programAction = ParseSyntaxOnly;
      break;
    case clang::driver::options::OPT_emit_mlir:
      opts.programAction = EmitMLIR;
      break;
    case clang::driver::options::OPT_emit_llvm:
      opts.programAction = EmitLLVM;
      break;
    case clang::driver::options::OPT_emit_llvm_bc:
      opts.programAction = EmitLLVMBitcode;
      break;
    case clang::driver::options::OPT_emit_obj:
      opts.programAction = EmitObj;
      break;
    case clang::driver::options::OPT_S:
      opts.programAction = EmitAssembly;
      break;
    case clang::driver::options::OPT_fdebug_unparse:
      opts.programAction = DebugUnparse;
      break;
    case clang::driver::options::OPT_fdebug_unparse_no_sema:
      opts.programAction = DebugUnparseNoSema;
      break;
    case clang::driver::options::OPT_fdebug_unparse_with_symbols:
      opts.programAction = DebugUnparseWithSymbols;
      break;
    case clang::driver::options::OPT_fdebug_dump_symbols:
      opts.programAction = DebugDumpSymbols;
      break;
    case clang::driver::options::OPT_fdebug_dump_parse_tree:
      opts.programAction = DebugDumpParseTree;
      break;
    case clang::driver::options::OPT_fdebug_dump_pft:
      opts.programAction = DebugDumpPFT;
      break;
    case clang::driver::options::OPT_fdebug_dump_all:
      opts.programAction = DebugDumpAll;
      break;
    case clang::driver::options::OPT_fdebug_dump_parse_tree_no_sema:
      opts.programAction = DebugDumpParseTreeNoSema;
      break;
    case clang::driver::options::OPT_fdebug_dump_provenance:
      opts.programAction = DebugDumpProvenance;
      break;
    case clang::driver::options::OPT_fdebug_dump_parsing_log:
      opts.programAction = DebugDumpParsingLog;
      break;
    case clang::driver::options::OPT_fdebug_measure_parse_tree:
      opts.programAction = DebugMeasureParseTree;
      break;
    case clang::driver::options::OPT_fdebug_pre_fir_tree:
      opts.programAction = DebugPreFIRTree;
      break;
    case clang::driver::options::OPT_fget_symbols_sources:
      opts.programAction = GetSymbolsSources;
      break;
    case clang::driver::options::OPT_fget_definition:
      opts.programAction = GetDefinition;
      break;
    case clang::driver::options::OPT_init_only:
      opts.programAction = InitOnly;
      break;

      // TODO:
      // case clang::driver::options::OPT_emit_llvm:
      // case clang::driver::options::OPT_emit_llvm_only:
      // case clang::driver::options::OPT_emit_codegen_only:
      // case clang::driver::options::OPT_emit_module:
      // (...)
    }

    // Parse the values provided with `-fget-definition` (there should be 3
    // integers)
    if (llvm::opt::OptSpecifier(a->getOption().getID()) ==
        clang::driver::options::OPT_fget_definition) {
      unsigned optVals[3] = {0, 0, 0};

      for (unsigned i = 0; i < 3; i++) {
        llvm::StringRef val = a->getValue(i);

        if (val.getAsInteger(10, optVals[i])) {
          // A non-integer was encountered - that's an error.
          diags.Report(clang::diag::err_drv_invalid_value)
              << a->getOption().getName() << val;
          break;
        }
      }
      opts.getDefVals.line = optVals[0];
      opts.getDefVals.startColumn = optVals[1];
      opts.getDefVals.endColumn = optVals[2];
    }
  }

  // Parsing -load <dsopath> option and storing shared object path
  if (llvm::opt::Arg *a = args.getLastArg(clang::driver::options::OPT_load)) {
    opts.plugins.push_back(a->getValue());
  }

  // Parsing -plugin <name> option and storing plugin name and setting action
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_plugin)) {
    opts.programAction = PluginAction;
    opts.actionName = a->getValue();
  }

  opts.outputFile = args.getLastArgValue(clang::driver::options::OPT_o);
  opts.showHelp = args.hasArg(clang::driver::options::OPT_help);
  opts.showVersion = args.hasArg(clang::driver::options::OPT_version);

  // Get the input kind (from the value passed via `-x`)
  InputKind dashX(Language::Unknown);
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_x)) {
    llvm::StringRef xValue = a->getValue();
    // Principal languages.
    dashX = llvm::StringSwitch<InputKind>(xValue)
                // Flang does not differentiate between pre-processed and not
                // pre-processed inputs.
                .Case("f95", Language::Fortran)
                .Case("f95-cpp-input", Language::Fortran)
                .Default(Language::Unknown);

    // Flang's intermediate representations.
    if (dashX.isUnknown())
      dashX = llvm::StringSwitch<InputKind>(xValue)
                  .Case("ir", Language::LLVM_IR)
                  .Case("fir", Language::MLIR)
                  .Case("mlir", Language::MLIR)
                  .Default(Language::Unknown);

    if (dashX.isUnknown())
      diags.Report(clang::diag::err_drv_invalid_value)
          << a->getAsString(args) << a->getValue();
  }

  // Collect the input files and save them in our instance of FrontendOptions.
  std::vector<std::string> inputs =
      args.getAllArgValues(clang::driver::options::OPT_INPUT);
  opts.inputs.clear();
  if (inputs.empty())
    // '-' is the default input if none is given.
    inputs.push_back("-");
  for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
    InputKind ik = dashX;
    if (ik.isUnknown()) {
      ik = FrontendOptions::getInputKindForExtension(
          llvm::StringRef(inputs[i]).rsplit('.').second);
      if (ik.isUnknown())
        ik = Language::Unknown;
      if (i == 0)
        dashX = ik;
    }

    opts.inputs.emplace_back(std::move(inputs[i]), ik);
  }

  // Set fortranForm based on options -ffree-form and -ffixed-form.
  if (const auto *arg = args.getLastArg(clang::driver::options::OPT_ffixed_form,
          clang::driver::options::OPT_ffree_form)) {
    opts.fortranForm =
        arg->getOption().matches(clang::driver::options::OPT_ffixed_form)
        ? FortranForm::FixedForm
        : FortranForm::FreeForm;
  }

  // Set fixedFormColumns based on -ffixed-line-length=<value>
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
      diags.Report(clang::diag::err_drv_negative_columns)
          << arg->getOption().getName() << arg->getValue();
    } else if (columns == 0) {
      opts.fixedFormColumns = 1000000;
    } else if (columns < 7) {
      diags.Report(clang::diag::err_drv_small_columns)
          << arg->getOption().getName() << arg->getValue() << "7";
    } else {
      opts.fixedFormColumns = columns;
    }
  }

  // Set conversion based on -fconvert=<value>
  if (const auto *arg =
          args.getLastArg(clang::driver::options::OPT_fconvert_EQ)) {
    const char *argValue = arg->getValue();
    if (auto convert = parseConvertArg(argValue))
      opts.envDefaults.push_back({"FORT_CONVERT", *convert});
    else
      diags.Report(clang::diag::err_drv_invalid_value)
          << arg->getAsString(args) << argValue;
  }

  // -f{no-}implicit-none
  opts.features.Enable(
      Fortran::common::LanguageFeature::ImplicitNoneTypeAlways,
      args.hasFlag(clang::driver::options::OPT_fimplicit_none,
          clang::driver::options::OPT_fno_implicit_none, false));

  // -f{no-}backslash
  opts.features.Enable(Fortran::common::LanguageFeature::BackslashEscapes,
      args.hasFlag(clang::driver::options::OPT_fbackslash,
          clang::driver::options::OPT_fno_backslash, false));

  // -f{no-}logical-abbreviations
  opts.features.Enable(Fortran::common::LanguageFeature::LogicalAbbreviations,
      args.hasFlag(clang::driver::options::OPT_flogical_abbreviations,
          clang::driver::options::OPT_fno_logical_abbreviations, false));

  // -f{no-}xor-operator
  opts.features.Enable(Fortran::common::LanguageFeature::XOROperator,
      args.hasFlag(clang::driver::options::OPT_fxor_operator,
          clang::driver::options::OPT_fno_xor_operator, false));

  // -fno-automatic
  if (args.hasArg(clang::driver::options::OPT_fno_automatic)) {
    opts.features.Enable(Fortran::common::LanguageFeature::DefaultSave);
  }

  if (args.hasArg(
          clang::driver::options::OPT_falternative_parameter_statement)) {
    opts.features.Enable(Fortran::common::LanguageFeature::OldStyleParameter);
  }
  if (const llvm::opt::Arg *arg =
          args.getLastArg(clang::driver::options::OPT_finput_charset_EQ)) {
    llvm::StringRef argValue = arg->getValue();
    if (argValue == "utf-8") {
      opts.encoding = Fortran::parser::Encoding::UTF_8;
    } else if (argValue == "latin-1") {
      opts.encoding = Fortran::parser::Encoding::LATIN_1;
    } else {
      diags.Report(clang::diag::err_drv_invalid_value)
          << arg->getAsString(args) << argValue;
    }
  }

  setUpFrontendBasedOnAction(opts);
  opts.dashX = dashX;

  return diags.getNumErrors() == numErrorsBefore;
}

// Generate the path to look for intrinsic modules
static std::string getIntrinsicDir() {
  // TODO: Find a system independent API
  llvm::SmallString<128> driverPath;
  driverPath.assign(llvm::sys::fs::getMainExecutable(nullptr, nullptr));
  llvm::sys::path::remove_filename(driverPath);
  driverPath.append("/../include/flang/");
  return std::string(driverPath);
}

// Generate the path to look for OpenMP headers
static std::string getOpenMPHeadersDir() {
  llvm::SmallString<128> includePath;
  includePath.assign(llvm::sys::fs::getMainExecutable(nullptr, nullptr));
  llvm::sys::path::remove_filename(includePath);
  includePath.append("/../include/flang/OpenMP/");
  return std::string(includePath);
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

  // Prepend the ordered list of -intrinsic-modules-path
  // to the default location to search.
  for (const auto *currentArg :
      args.filtered(clang::driver::options::OPT_fintrinsic_modules_path))
    opts.searchDirectoriesFromIntrModPath.emplace_back(currentArg->getValue());

  // -cpp/-nocpp
  if (const auto *currentArg = args.getLastArg(
          clang::driver::options::OPT_cpp, clang::driver::options::OPT_nocpp))
    opts.macrosFlag =
        (currentArg->getOption().matches(clang::driver::options::OPT_cpp))
        ? PPMacrosFlag::Include
        : PPMacrosFlag::Exclude;

  opts.noReformat = args.hasArg(clang::driver::options::OPT_fno_reformat);
  opts.noLineDirectives = args.hasArg(clang::driver::options::OPT_P);
}

/// Parses all semantic related arguments and populates the variables
/// options accordingly. Returns false if new errors are generated.
static bool parseSemaArgs(CompilerInvocation &res, llvm::opt::ArgList &args,
    clang::DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // -J/module-dir option
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
    res.setModuleDir(moduleDirList[0]);

  // -fdebug-module-writer option
  if (args.hasArg(clang::driver::options::OPT_fdebug_module_writer)) {
    res.setDebugModuleDir(true);
  }

  // -module-suffix
  if (const auto *moduleSuffix =
          args.getLastArg(clang::driver::options::OPT_module_suffix)) {
    res.setModuleFileSuffix(moduleSuffix->getValue());
  }

  // -f{no-}analyzed-objects-for-unparse
  res.setUseAnalyzedObjectsForUnparse(args.hasFlag(
      clang::driver::options::OPT_fanalyzed_objects_for_unparse,
      clang::driver::options::OPT_fno_analyzed_objects_for_unparse, true));

  return diags.getNumErrors() == numErrorsBefore;
}

/// Parses all diagnostics related arguments and populates the variables
/// options accordingly. Returns false if new errors are generated.
static bool parseDiagArgs(CompilerInvocation &res, llvm::opt::ArgList &args,
    clang::DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // -Werror option
  // TODO: Currently throws a Diagnostic for anything other than -W<error>,
  // this has to change when other -W<opt>'s are supported.
  if (args.hasArg(clang::driver::options::OPT_W_Joined)) {
    if (args.getLastArgValue(clang::driver::options::OPT_W_Joined)
            .equals("error")) {
      res.setWarnAsErr(true);
    } else {
      const unsigned diagID =
          diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
              "Only `-Werror` is supported currently.");
      diags.Report(diagID);
    }
  }

  // Default to off for `flang-new -fc1`.
  res.getFrontendOpts().showColors =
      parseShowColorsArgs(args, /*defaultDiagColor=*/false);

  return diags.getNumErrors() == numErrorsBefore;
}

/// Parses all Dialect related arguments and populates the variables
/// options accordingly. Returns false if new errors are generated.
static bool parseDialectArgs(CompilerInvocation &res, llvm::opt::ArgList &args,
    clang::DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // -fdefault* family
  if (args.hasArg(clang::driver::options::OPT_fdefault_real_8)) {
    res.getDefaultKinds().set_defaultRealKind(8);
    res.getDefaultKinds().set_doublePrecisionKind(16);
  }
  if (args.hasArg(clang::driver::options::OPT_fdefault_integer_8)) {
    res.getDefaultKinds().set_defaultIntegerKind(8);
    res.getDefaultKinds().set_subscriptIntegerKind(8);
    res.getDefaultKinds().set_sizeIntegerKind(8);
  }
  if (args.hasArg(clang::driver::options::OPT_fdefault_double_8)) {
    if (!args.hasArg(clang::driver::options::OPT_fdefault_real_8)) {
      // -fdefault-double-8 has to be used with -fdefault-real-8
      // to be compatible with gfortran
      const unsigned diagID =
          diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
              "Use of `-fdefault-double-8` requires `-fdefault-real-8`");
      diags.Report(diagID);
    }
    // https://gcc.gnu.org/onlinedocs/gfortran/Fortran-Dialect-Options.html
    res.getDefaultKinds().set_doublePrecisionKind(8);
  }
  if (args.hasArg(clang::driver::options::OPT_flarge_sizes))
    res.getDefaultKinds().set_sizeIntegerKind(8);

  // -fopenmp and -fopenacc
  if (args.hasArg(clang::driver::options::OPT_fopenacc)) {
    res.getFrontendOpts().features.Enable(
        Fortran::common::LanguageFeature::OpenACC);
  }
  if (args.hasArg(clang::driver::options::OPT_fopenmp)) {
    res.getFrontendOpts().features.Enable(
        Fortran::common::LanguageFeature::OpenMP);
  }

  // -pedantic
  if (args.hasArg(clang::driver::options::OPT_pedantic)) {
    res.setEnableConformanceChecks();
  }
  // -std=f2018 (currently this implies -pedantic)
  // TODO: Set proper options when more fortran standards
  // are supported.
  if (args.hasArg(clang::driver::options::OPT_std_EQ)) {
    auto standard = args.getLastArgValue(clang::driver::options::OPT_std_EQ);
    // We only allow f2018 as the given standard
    if (standard.equals("f2018")) {
      res.setEnableConformanceChecks();
    } else {
      const unsigned diagID =
          diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
              "Only -std=f2018 is allowed currently.");
      diags.Report(diagID);
    }
  }
  return diags.getNumErrors() == numErrorsBefore;
}

bool CompilerInvocation::createFromArgs(
    CompilerInvocation &res, llvm::ArrayRef<const char *> commandLineArgs,
    clang::DiagnosticsEngine &diags) {

  bool success = true;

  // Set the default triple for this CompilerInvocation. This might be
  // overridden by users with `-triple` (see the call to `ParseTargetArgs`
  // below).
  // NOTE: Like in Clang, it would be nice to use option marshalling
  // for this so that the entire logic for setting-up the triple is in one
  // place.
  res.getTargetOpts().triple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());

  // Parse the arguments
  const llvm::opt::OptTable &opts = clang::driver::getDriverOptTable();
  const unsigned includedFlagsBitmask = clang::driver::options::FC1Option;
  unsigned missingArgIndex, missingArgCount;
  llvm::opt::InputArgList args = opts.ParseArgs(
      commandLineArgs, missingArgIndex, missingArgCount, includedFlagsBitmask);

  // Check for missing argument error.
  if (missingArgCount) {
    diags.Report(clang::diag::err_drv_missing_argument)
        << args.getArgString(missingArgIndex) << missingArgCount;
    success = false;
  }

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

  success &= parseFrontendArgs(res.getFrontendOpts(), args, diags);
  parseTargetArgs(res.getTargetOpts(), args);
  parsePreprocessorArgs(res.getPreprocessorOpts(), args);
  parseCodeGenArgs(res.getCodeGenOpts(), args, diags);
  success &= parseSemaArgs(res, args, diags);
  success &= parseDialectArgs(res, args, diags);
  success &= parseDiagArgs(res, args, diags);
  res.frontendOpts.llvmArgs =
      args.getAllArgValues(clang::driver::options::OPT_mllvm);

  res.frontendOpts.mlirArgs =
      args.getAllArgValues(clang::driver::options::OPT_mmlir);

  return success;
}

void CompilerInvocation::collectMacroDefinitions() {
  auto &ppOpts = this->getPreprocessorOpts();

  for (unsigned i = 0, n = ppOpts.macros.size(); i != n; ++i) {
    llvm::StringRef macro = ppOpts.macros[i].first;
    bool isUndef = ppOpts.macros[i].second;

    std::pair<llvm::StringRef, llvm::StringRef> macroPair = macro.split('=');
    llvm::StringRef macroName = macroPair.first;
    llvm::StringRef macroBody = macroPair.second;

    // For an #undef'd macro, we only care about the name.
    if (isUndef) {
      parserOpts.predefinitions.emplace_back(macroName.str(),
                                             std::optional<std::string>{});
      continue;
    }

    // For a #define'd macro, figure out the actual definition.
    if (macroName.size() == macro.size())
      macroBody = "1";
    else {
      // Note: GCC drops anything following an end-of-line character.
      llvm::StringRef::size_type end = macroBody.find_first_of("\n\r");
      macroBody = macroBody.substr(0, end);
    }
    parserOpts.predefinitions.emplace_back(
        macroName, std::optional<std::string>(macroBody.str()));
  }
}

void CompilerInvocation::setDefaultFortranOpts() {
  auto &fortranOptions = getFortranOpts();

  std::vector<std::string> searchDirectories{"."s};
  fortranOptions.searchDirectories = searchDirectories;

  // Add the location of omp_lib.h to the search directories. Currently this is
  // identical to the modules' directory.
  fortranOptions.searchDirectories.emplace_back(getOpenMPHeadersDir());

  fortranOptions.isFixedForm = false;
}

// TODO: When expanding this method, consider creating a dedicated API for
// this. Also at some point we will need to differentiate between different
// targets and add dedicated predefines for each.
void CompilerInvocation::setDefaultPredefinitions() {
  auto &fortranOptions = getFortranOpts();
  const auto &frontendOptions = getFrontendOpts();

  // Populate the macro list with version numbers and other predefinitions.
  fortranOptions.predefinitions.emplace_back("__flang__", "1");
  fortranOptions.predefinitions.emplace_back(
      "__flang_major__", FLANG_VERSION_MAJOR_STRING);
  fortranOptions.predefinitions.emplace_back(
      "__flang_minor__", FLANG_VERSION_MINOR_STRING);
  fortranOptions.predefinitions.emplace_back(
      "__flang_patchlevel__", FLANG_VERSION_PATCHLEVEL_STRING);

  // Add predefinitions based on extensions enabled
  if (frontendOptions.features.IsEnabled(
          Fortran::common::LanguageFeature::OpenACC)) {
    fortranOptions.predefinitions.emplace_back("_OPENACC", "202011");
  }
  if (frontendOptions.features.IsEnabled(
          Fortran::common::LanguageFeature::OpenMP)) {
    fortranOptions.predefinitions.emplace_back("_OPENMP", "201511");
  }
  llvm::Triple targetTriple{llvm::Triple(this->targetOpts.triple)};
  if (targetTriple.getArch() == llvm::Triple::ArchType::x86_64) {
    fortranOptions.predefinitions.emplace_back("__x86_64__", "1");
    fortranOptions.predefinitions.emplace_back("__x86_64", "1");
  }
}

void CompilerInvocation::setFortranOpts() {
  auto &fortranOptions = getFortranOpts();
  const auto &frontendOptions = getFrontendOpts();
  const auto &preprocessorOptions = getPreprocessorOpts();
  auto &moduleDirJ = getModuleDir();

  if (frontendOptions.fortranForm != FortranForm::Unknown) {
    fortranOptions.isFixedForm =
        frontendOptions.fortranForm == FortranForm::FixedForm;
  }
  fortranOptions.fixedFormColumns = frontendOptions.fixedFormColumns;

  fortranOptions.features = frontendOptions.features;
  fortranOptions.encoding = frontendOptions.encoding;

  // Adding search directories specified by -I
  fortranOptions.searchDirectories.insert(
      fortranOptions.searchDirectories.end(),
      preprocessorOptions.searchDirectoriesFromDashI.begin(),
      preprocessorOptions.searchDirectoriesFromDashI.end());

  // Add the ordered list of -intrinsic-modules-path
  fortranOptions.searchDirectories.insert(
      fortranOptions.searchDirectories.end(),
      preprocessorOptions.searchDirectoriesFromIntrModPath.begin(),
      preprocessorOptions.searchDirectoriesFromIntrModPath.end());

  //  Add the default intrinsic module directory
  fortranOptions.intrinsicModuleDirectories.emplace_back(getIntrinsicDir());

  // Add the directory supplied through -J/-module-dir to the list of search
  // directories
  if (moduleDirJ != ".")
    fortranOptions.searchDirectories.emplace_back(moduleDirJ);

  if (frontendOptions.instrumentedParse)
    fortranOptions.instrumentedParse = true;

  if (frontendOptions.showColors)
    fortranOptions.showColors = true;

  if (frontendOptions.needProvenanceRangeToCharBlockMappings)
    fortranOptions.needProvenanceRangeToCharBlockMappings = true;

  if (getEnableConformanceChecks()) {
    fortranOptions.features.WarnOnAllNonstandard();
  }
}

void CompilerInvocation::setSemanticsOpts(
    Fortran::parser::AllCookedSources &allCookedSources) {
  auto &fortranOptions = getFortranOpts();

  semanticsContext = std::make_unique<semantics::SemanticsContext>(
      getDefaultKinds(), fortranOptions.features, allCookedSources);

  semanticsContext->set_moduleDirectory(getModuleDir())
      .set_searchDirectories(fortranOptions.searchDirectories)
      .set_intrinsicModuleDirectories(fortranOptions.intrinsicModuleDirectories)
      .set_warnOnNonstandardUsage(getEnableConformanceChecks())
      .set_warningsAreErrors(getWarnAsErr())
      .set_moduleFileSuffix(getModuleFileSuffix());

  llvm::Triple targetTriple{llvm::Triple(this->targetOpts.triple)};
  // FIXME: Handle real(3) ?
  if (targetTriple.getArch() != llvm::Triple::ArchType::x86_64) {
    semanticsContext->targetCharacteristics().DisableType(
        Fortran::common::TypeCategory::Real, /*kind=*/10);
  }
}

/// Set \p loweringOptions controlling lowering behavior based
/// on the \p optimizationLevel.
void CompilerInvocation::setLoweringOptions() {
  const auto &codegenOpts = getCodeGenOpts();

  // Lower TRANSPOSE as a runtime call under -O0.
  loweringOpts.setOptimizeTranspose(codegenOpts.OptimizationLevel > 0);
}
