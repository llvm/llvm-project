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
#include "flang/Frontend/CodeGenOptions.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Frontend/TargetOptions.h"
#include "flang/Optimizer/Passes/CommandLineOpts.h"
#include "flang/Semantics/semantics.h"
#include "flang/Support/Fortran-features.h"
#include "flang/Support/OpenMP-features.h"
#include "flang/Support/Version.h"
#include "flang/Tools/TargetSetup.h"
#include "flang/Version.inc"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/CommonArgs.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/OptionUtils.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Frontend/Debug/Options.h"
#include "llvm/Frontend/Driver/CodeGenOptions.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>

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
  // compiler driver `flang` but default to off in the frontend driver
  // `flang -fc1`, needing an explicit OPT_fdiagnostics_color.
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
  unsigned defaultOpt = 0;

  if (llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_O_Group)) {
    if (a->getOption().matches(clang::driver::options::OPT_O0))
      return 0;

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

static bool parseDebugArgs(Fortran::frontend::CodeGenOptions &opts,
                           llvm::opt::ArgList &args,
                           clang::DiagnosticsEngine &diags) {
  using DebugInfoKind = llvm::codegenoptions::DebugInfoKind;
  if (llvm::opt::Arg *arg =
          args.getLastArg(clang::driver::options::OPT_debug_info_kind_EQ)) {
    std::optional<DebugInfoKind> val =
        llvm::StringSwitch<std::optional<DebugInfoKind>>(arg->getValue())
            .Case("line-tables-only", llvm::codegenoptions::DebugLineTablesOnly)
            .Case("line-directives-only",
                  llvm::codegenoptions::DebugDirectivesOnly)
            .Case("constructor", llvm::codegenoptions::DebugInfoConstructor)
            .Case("limited", llvm::codegenoptions::LimitedDebugInfo)
            .Case("standalone", llvm::codegenoptions::FullDebugInfo)
            .Case("unused-types", llvm::codegenoptions::UnusedTypeInfo)
            .Default(std::nullopt);
    if (!val.has_value()) {
      diags.Report(clang::diag::err_drv_invalid_value)
          << arg->getAsString(args) << arg->getValue();
      return false;
    }
    opts.setDebugInfo(val.value());
    if (val != llvm::codegenoptions::DebugLineTablesOnly &&
        val != llvm::codegenoptions::FullDebugInfo &&
        val != llvm::codegenoptions::NoDebugInfo) {
      const auto debugWarning = diags.getCustomDiagID(
          clang::DiagnosticsEngine::Warning, "Unsupported debug option: %0");
      diags.Report(debugWarning) << arg->getValue();
    }
  }
  return true;
}

static void parseDoConcurrentMapping(Fortran::frontend::CodeGenOptions &opts,
                                     llvm::opt::ArgList &args,
                                     clang::DiagnosticsEngine &diags) {
  llvm::opt::Arg *arg =
      args.getLastArg(clang::driver::options::OPT_fdo_concurrent_to_openmp_EQ);
  if (!arg)
    return;

  using DoConcurrentMappingKind =
      Fortran::frontend::CodeGenOptions::DoConcurrentMappingKind;
  std::optional<DoConcurrentMappingKind> val =
      llvm::StringSwitch<std::optional<DoConcurrentMappingKind>>(
          arg->getValue())
          .Case("none", DoConcurrentMappingKind::DCMK_None)
          .Case("host", DoConcurrentMappingKind::DCMK_Host)
          .Case("device", DoConcurrentMappingKind::DCMK_Device)
          .Default(std::nullopt);

  if (!val.has_value()) {
    diags.Report(clang::diag::err_drv_invalid_value)
        << arg->getAsString(args) << arg->getValue();
  }

  opts.setDoConcurrentMapping(val.value());
}

static bool parseVectorLibArg(Fortran::frontend::CodeGenOptions &opts,
                              llvm::opt::ArgList &args,
                              clang::DiagnosticsEngine &diags) {
  llvm::opt::Arg *arg = args.getLastArg(clang::driver::options::OPT_fveclib);
  if (!arg)
    return true;

  using VectorLibrary = llvm::driver::VectorLibrary;
  std::optional<VectorLibrary> val =
      llvm::StringSwitch<std::optional<VectorLibrary>>(arg->getValue())
          .Case("Accelerate", VectorLibrary::Accelerate)
          .Case("libmvec", VectorLibrary::LIBMVEC)
          .Case("MASSV", VectorLibrary::MASSV)
          .Case("SVML", VectorLibrary::SVML)
          .Case("SLEEF", VectorLibrary::SLEEF)
          .Case("Darwin_libsystem_m", VectorLibrary::Darwin_libsystem_m)
          .Case("ArmPL", VectorLibrary::ArmPL)
          .Case("AMDLIBM", VectorLibrary::AMDLIBM)
          .Case("NoLibrary", VectorLibrary::NoLibrary)
          .Default(std::nullopt);
  if (!val.has_value()) {
    diags.Report(clang::diag::err_drv_invalid_value)
        << arg->getAsString(args) << arg->getValue();
    return false;
  }
  opts.setVecLib(val.value());
  return true;
}

// Generate an OptRemark object containing info on if the -Rgroup
// specified is enabled or not.
static CodeGenOptions::OptRemark
parseOptimizationRemark(clang::DiagnosticsEngine &diags,
                        llvm::opt::ArgList &args, llvm::opt::OptSpecifier optEq,
                        llvm::StringRef remarkOptName) {
  assert((remarkOptName == "pass" || remarkOptName == "pass-missed" ||
          remarkOptName == "pass-analysis") &&
         "Unsupported remark option name provided.");
  CodeGenOptions::OptRemark result;

  for (llvm::opt::Arg *a : args) {
    if (a->getOption().matches(clang::driver::options::OPT_R_Joined)) {
      llvm::StringRef value = a->getValue();

      if (value == remarkOptName) {
        result.Kind = CodeGenOptions::RemarkKind::RK_Enabled;
        // Enable everything
        result.Pattern = ".*";
        result.Regex = std::make_shared<llvm::Regex>(result.Pattern);

      } else if (value.split('-') ==
                 std::make_pair(llvm::StringRef("no"), remarkOptName)) {
        result.Kind = CodeGenOptions::RemarkKind::RK_Disabled;
        // Disable everything
        result.Pattern = "";
        result.Regex = nullptr;
      }
    } else if (a->getOption().matches(optEq)) {
      result.Kind = CodeGenOptions::RemarkKind::RK_WithPattern;
      result.Pattern = a->getValue();
      result.Regex = std::make_shared<llvm::Regex>(result.Pattern);
      std::string regexError;

      if (!result.Regex->isValid(regexError)) {
        diags.Report(clang::diag::err_drv_optimization_remark_pattern)
            << regexError << a->getAsString(args);
        return CodeGenOptions::OptRemark();
      }
    }
  }
  return result;
}

static void parseCodeGenArgs(Fortran::frontend::CodeGenOptions &opts,
                             llvm::opt::ArgList &args,
                             clang::DiagnosticsEngine &diags) {
  opts.OptimizationLevel = getOptimizationLevel(args, diags);

  if (args.hasFlag(clang::driver::options::OPT_fdebug_pass_manager,
                   clang::driver::options::OPT_fno_debug_pass_manager, false))
    opts.DebugPassManager = 1;

  if (args.hasFlag(clang::driver::options::OPT_fstack_arrays,
                   clang::driver::options::OPT_fno_stack_arrays, false))
    opts.StackArrays = 1;

  if (args.getLastArg(clang::driver::options::OPT_floop_interchange))
    opts.InterchangeLoops = 1;

  if (args.getLastArg(clang::driver::options::OPT_vectorize_loops))
    opts.VectorizeLoop = 1;

  if (args.getLastArg(clang::driver::options::OPT_vectorize_slp))
    opts.VectorizeSLP = 1;

  if (args.hasFlag(clang::driver::options::OPT_floop_versioning,
                   clang::driver::options::OPT_fno_loop_versioning, false))
    opts.LoopVersioning = 1;

  opts.UnrollLoops = args.hasFlag(clang::driver::options::OPT_funroll_loops,
                                  clang::driver::options::OPT_fno_unroll_loops,
                                  (opts.OptimizationLevel > 1));

  opts.AliasAnalysis = opts.OptimizationLevel > 0;

  // -mframe-pointer=none/non-leaf/reserved/all option.
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_mframe_pointer_EQ)) {
    std::optional<llvm::FramePointerKind> val =
        llvm::StringSwitch<std::optional<llvm::FramePointerKind>>(a->getValue())
            .Case("none", llvm::FramePointerKind::None)
            .Case("non-leaf", llvm::FramePointerKind::NonLeaf)
            .Case("reserved", llvm::FramePointerKind::Reserved)
            .Case("all", llvm::FramePointerKind::All)
            .Default(std::nullopt);

    if (!val.has_value()) {
      diags.Report(clang::diag::err_drv_invalid_value)
          << a->getAsString(args) << a->getValue();
    } else
      opts.setFramePointer(val.value());
  }

  for (auto *a : args.filtered(clang::driver::options::OPT_fpass_plugin_EQ))
    opts.LLVMPassPlugins.push_back(a->getValue());

  opts.Reciprocals = clang::driver::tools::parseMRecipOption(diags, args);

  opts.PreferVectorWidth =
      clang::driver::tools::parseMPreferVectorWidthOption(diags, args);

  // -fembed-offload-object option
  for (auto *a :
       args.filtered(clang::driver::options::OPT_fembed_offload_object_EQ))
    opts.OffloadObjects.push_back(a->getValue());

  if (args.hasArg(clang::driver::options::OPT_finstrument_functions))
    opts.InstrumentFunctions = 1;

  // -flto=full/thin option.
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_flto_EQ)) {
    llvm::StringRef s = a->getValue();
    assert((s == "full" || s == "thin") && "Unknown LTO mode.");
    if (s == "full")
      opts.PrepareForFullLTO = true;
    else
      opts.PrepareForThinLTO = true;
  }

  if (const llvm::opt::Arg *a = args.getLastArg(
          clang::driver::options::OPT_mcode_object_version_EQ)) {
    llvm::StringRef s = a->getValue();
    if (s == "6")
      opts.CodeObjectVersion = llvm::CodeObjectVersionKind::COV_6;
    if (s == "5")
      opts.CodeObjectVersion = llvm::CodeObjectVersionKind::COV_5;
    if (s == "4")
      opts.CodeObjectVersion = llvm::CodeObjectVersionKind::COV_4;
    if (s == "none")
      opts.CodeObjectVersion = llvm::CodeObjectVersionKind::COV_None;
  }

  // -f[no-]save-optimization-record[=<format>]
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_opt_record_file))
    opts.OptRecordFile = a->getValue();

  // Optimization file format. Defaults to yaml
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_opt_record_format))
    opts.OptRecordFormat = a->getValue();

  // Specifies, using a regex, which successful optimization passes(middle and
  // backend), to include in the final optimization record file generated. If
  // not provided -fsave-optimization-record will include all passes.
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_opt_record_passes))
    opts.OptRecordPasses = a->getValue();

  // Create OptRemark that allows printing of all successful optimization
  // passes applied.
  opts.OptimizationRemark =
      parseOptimizationRemark(diags, args, clang::driver::options::OPT_Rpass_EQ,
                              /*remarkOptName=*/"pass");

  // Create OptRemark that allows all missed optimization passes to be printed.
  opts.OptimizationRemarkMissed = parseOptimizationRemark(
      diags, args, clang::driver::options::OPT_Rpass_missed_EQ,
      /*remarkOptName=*/"pass-missed");

  // Create OptRemark that allows all optimization decisions made by LLVM
  // to be printed.
  opts.OptimizationRemarkAnalysis = parseOptimizationRemark(
      diags, args, clang::driver::options::OPT_Rpass_analysis_EQ,
      /*remarkOptName=*/"pass-analysis");

  if (opts.getDebugInfo() == llvm::codegenoptions::NoDebugInfo) {
    // If the user requested a flag that requires source locations available in
    // the backend, make sure that the backend tracks source location
    // information.
    bool needLocTracking = !opts.OptRecordFile.empty() ||
                           !opts.OptRecordPasses.empty() ||
                           !opts.OptRecordFormat.empty() ||
                           opts.OptimizationRemark.hasValidPattern() ||
                           opts.OptimizationRemarkMissed.hasValidPattern() ||
                           opts.OptimizationRemarkAnalysis.hasValidPattern();

    if (needLocTracking)
      opts.setDebugInfo(llvm::codegenoptions::LocTrackingOnly);
  }

  if (auto *a = args.getLastArg(clang::driver::options::OPT_save_temps_EQ))
    opts.SaveTempsDir = a->getValue();

  // -record-command-line option.
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_record_command_line)) {
    opts.RecordCommandLine = a->getValue();
  }

  // -mlink-builtin-bitcode
  for (auto *a :
       args.filtered(clang::driver::options::OPT_mlink_builtin_bitcode))
    opts.BuiltinBCLibs.push_back(a->getValue());

  // -mrelocation-model option.
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_mrelocation_model)) {
    llvm::StringRef modelName = a->getValue();
    auto relocModel =
        llvm::StringSwitch<std::optional<llvm::Reloc::Model>>(modelName)
            .Case("static", llvm::Reloc::Static)
            .Case("pic", llvm::Reloc::PIC_)
            .Case("dynamic-no-pic", llvm::Reloc::DynamicNoPIC)
            .Case("ropi", llvm::Reloc::ROPI)
            .Case("rwpi", llvm::Reloc::RWPI)
            .Case("ropi-rwpi", llvm::Reloc::ROPI_RWPI)
            .Default(std::nullopt);
    if (relocModel.has_value())
      opts.setRelocationModel(*relocModel);
    else
      diags.Report(clang::diag::err_drv_invalid_value)
          << a->getAsString(args) << modelName;
  }

  // -pic-level and -pic-is-pie option.
  if (int picLevel = getLastArgIntValue(
          args, clang::driver::options::OPT_pic_level, 0, diags)) {
    if (picLevel > 2)
      diags.Report(clang::diag::err_drv_invalid_value)
          << args.getLastArg(clang::driver::options::OPT_pic_level)
                 ->getAsString(args)
          << picLevel;

    opts.PICLevel = picLevel;
    if (args.hasArg(clang::driver::options::OPT_pic_is_pie))
      opts.IsPIE = 1;
  }

  if (args.hasArg(clang::driver::options::OPT_fprofile_generate)) {
    opts.setProfileInstr(llvm::driver::ProfileInstrKind::ProfileIRInstr);
  }

  if (auto A = args.getLastArg(clang::driver::options::OPT_fprofile_use_EQ)) {
    opts.setProfileUse(llvm::driver::ProfileInstrKind::ProfileIRInstr);
    opts.ProfileInstrumentUsePath = A->getValue();
  }

  // -mcmodel option.
  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_mcmodel_EQ)) {
    llvm::StringRef modelName = a->getValue();
    std::optional<llvm::CodeModel::Model> codeModel = getCodeModel(modelName);

    if (codeModel.has_value())
      opts.CodeModel = modelName;
    else
      diags.Report(clang::diag::err_drv_invalid_value)
          << a->getAsString(args) << modelName;
  }

  if (const llvm::opt::Arg *arg = args.getLastArg(
          clang::driver::options::OPT_mlarge_data_threshold_EQ)) {
    uint64_t LDT;
    if (llvm::StringRef(arg->getValue()).getAsInteger(/*Radix=*/10, LDT)) {
      diags.Report(clang::diag::err_drv_invalid_value)
          << arg->getSpelling() << arg->getValue();
    }
    opts.LargeDataThreshold = LDT;
  }

  // This option is compatible with -f[no-]underscoring in gfortran.
  if (args.hasFlag(clang::driver::options::OPT_fno_underscoring,
                   clang::driver::options::OPT_funderscoring, false)) {
    opts.Underscoring = 0;
  }

  parseDoConcurrentMapping(opts, args, diags);

  if (const llvm::opt::Arg *arg =
          args.getLastArg(clang::driver::options::OPT_complex_range_EQ)) {
    llvm::StringRef argValue = llvm::StringRef(arg->getValue());
    if (argValue == "full") {
      opts.setComplexRange(CodeGenOptions::ComplexRangeKind::CX_Full);
    } else if (argValue == "improved") {
      opts.setComplexRange(CodeGenOptions::ComplexRangeKind::CX_Improved);
    } else if (argValue == "basic") {
      opts.setComplexRange(CodeGenOptions::ComplexRangeKind::CX_Basic);
    } else {
      diags.Report(clang::diag::err_drv_invalid_value)
          << arg->getAsString(args) << arg->getValue();
    }
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

  opts.atomicIgnoreDenormalMode = args.hasFlag(
      clang::driver::options::OPT_fatomic_ignore_denormal_mode,
      clang::driver::options::OPT_fno_atomic_ignore_denormal_mode, false);
  opts.atomicFineGrainedMemory = args.hasFlag(
      clang::driver::options::OPT_fatomic_fine_grained_memory,
      clang::driver::options::OPT_fno_atomic_fine_grained_memory, false);
  opts.atomicRemoteMemory =
      args.hasFlag(clang::driver::options::OPT_fatomic_remote_memory,
                   clang::driver::options::OPT_fno_atomic_remote_memory, false);

  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_target_cpu))
    opts.cpu = a->getValue();

  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_tune_cpu))
    opts.cpuToTuneFor = a->getValue();

  for (const llvm::opt::Arg *currentArg :
       args.filtered(clang::driver::options::OPT_target_feature))
    opts.featuresAsWritten.emplace_back(currentArg->getValue());

  if (args.hasArg(clang::driver::options::OPT_fdisable_real_10))
    opts.disabledRealKinds.push_back(10);

  if (args.hasArg(clang::driver::options::OPT_fdisable_real_3))
    opts.disabledRealKinds.push_back(3);

  if (args.hasArg(clang::driver::options::OPT_fdisable_integer_2))
    opts.disabledIntegerKinds.push_back(2);

  if (args.hasArg(clang::driver::options::OPT_fdisable_integer_16))
    opts.disabledIntegerKinds.push_back(16);

  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_mabi_EQ)) {
    opts.abi = a->getValue();
    llvm::StringRef V = a->getValue();
    if (V == "vec-extabi") {
      opts.EnableAIXExtendedAltivecABI = true;
    } else if (V == "vec-default") {
      opts.EnableAIXExtendedAltivecABI = false;
    }
  }

  opts.asmVerbose =
      args.hasFlag(clang::driver::options::OPT_fverbose_asm,
                   clang::driver::options::OPT_fno_verbose_asm, false);
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
    case clang::driver::options::OPT_emit_fir:
      opts.programAction = EmitFIR;
      break;
    case clang::driver::options::OPT_emit_hlfir:
      opts.programAction = EmitHLFIR;
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
    case clang::driver::options::OPT_fdebug_unparse_with_modules:
      opts.programAction = DebugUnparseWithModules;
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
  opts.printSupportedCPUs =
      args.hasArg(clang::driver::options::OPT_print_supported_cpus);

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
                // CUDA Fortran
                .Case("cuda", Language::Fortran)
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
  if (const auto *arg =
          args.getLastArg(clang::driver::options::OPT_ffixed_form,
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

  // -f{no-}implicit-none-ext
  opts.features.Enable(
      Fortran::common::LanguageFeature::ImplicitNoneExternal,
      args.hasFlag(clang::driver::options::OPT_fimplicit_none_ext,
                   clang::driver::options::OPT_fno_implicit_none_ext, false));

  // -f{no-}backslash
  opts.features.Enable(Fortran::common::LanguageFeature::BackslashEscapes,
                       args.hasFlag(clang::driver::options::OPT_fbackslash,
                                    clang::driver::options::OPT_fno_backslash,
                                    false));

  // -f{no-}logical-abbreviations
  opts.features.Enable(
      Fortran::common::LanguageFeature::LogicalAbbreviations,
      args.hasFlag(clang::driver::options::OPT_flogical_abbreviations,
                   clang::driver::options::OPT_fno_logical_abbreviations,
                   false));

  // -f{no-}unsigned
  opts.features.Enable(Fortran::common::LanguageFeature::Unsigned,
                       args.hasFlag(clang::driver::options::OPT_funsigned,
                                    clang::driver::options::OPT_fno_unsigned,
                                    false));

  // -f{no-}xor-operator
  opts.features.Enable(
      Fortran::common::LanguageFeature::XOROperator,
      args.hasFlag(clang::driver::options::OPT_fxor_operator,
                   clang::driver::options::OPT_fno_xor_operator, false));

  // -fno-automatic
  if (args.hasArg(clang::driver::options::OPT_fno_automatic)) {
    opts.features.Enable(Fortran::common::LanguageFeature::DefaultSave);
  }

  // -f{no}-save-main-program
  opts.features.Enable(
      Fortran::common::LanguageFeature::SaveMainProgram,
      args.hasFlag(clang::driver::options::OPT_fsave_main_program,
                   clang::driver::options::OPT_fno_save_main_program, false));

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
static std::string getIntrinsicDir(const char *argv) {
  // TODO: Find a system independent API
  llvm::SmallString<128> driverPath;
  driverPath.assign(llvm::sys::fs::getMainExecutable(argv, nullptr));
  llvm::sys::path::remove_filename(driverPath);
  driverPath.append("/../include/flang/");
  return std::string(driverPath);
}

// Generate the path to look for OpenMP headers
static std::string getOpenMPHeadersDir(const char *argv) {
  llvm::SmallString<128> includePath;
  includePath.assign(llvm::sys::fs::getMainExecutable(argv, nullptr));
  llvm::sys::path::remove_filename(includePath);
  includePath.append("/../include/flang/OpenMP/");
  return std::string(includePath);
}

/// Parses all preprocessor input arguments and populates the preprocessor
/// options accordingly.
///
/// \param [in] opts The preprocessor options instance
/// \param [out] args The list of input arguments
static void parsePreprocessorArgs(Fortran::frontend::PreprocessorOptions &opts,
                                  llvm::opt::ArgList &args) {
  // Add macros from the command line.
  for (const auto *currentArg : args.filtered(clang::driver::options::OPT_D,
                                              clang::driver::options::OPT_U)) {
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
  // Enable -cpp based on -x unless explicitly disabled with -nocpp
  if (opts.macrosFlag != PPMacrosFlag::Exclude)
    if (const auto *dashX = args.getLastArg(clang::driver::options::OPT_x))
      opts.macrosFlag = llvm::StringSwitch<PPMacrosFlag>(dashX->getValue())
                            .Case("f95-cpp-input", PPMacrosFlag::Include)
                            .Default(opts.macrosFlag);

  opts.noReformat = args.hasArg(clang::driver::options::OPT_fno_reformat);
  opts.preprocessIncludeLines =
      args.hasArg(clang::driver::options::OPT_fpreprocess_include_lines);
  opts.noLineDirectives = args.hasArg(clang::driver::options::OPT_P);
  opts.showMacros = args.hasArg(clang::driver::options::OPT_dM);
}

/// Parses all semantic related arguments and populates the variables
/// options accordingly. Returns false if new errors are generated.
static bool parseSemaArgs(CompilerInvocation &res, llvm::opt::ArgList &args,
                          clang::DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // -J/module-dir option
  std::vector<std::string> moduleDirList =
      args.getAllArgValues(clang::driver::options::OPT_module_dir);
  // User can only specify one -J/-module-dir directory, but may repeat
  // -J/-module-dir as long as the directory is the same each time.
  // https://gcc.gnu.org/onlinedocs/gfortran/Directory-Options.html
  std::sort(moduleDirList.begin(), moduleDirList.end());
  moduleDirList.erase(std::unique(moduleDirList.begin(), moduleDirList.end()),
                      moduleDirList.end());
  if (moduleDirList.size() > 1) {
    const unsigned diagID =
        diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                              "Only one '-module-dir/-J' directory allowed. "
                              "'-module-dir/-J' may be given multiple times "
                              "but the directory must be the same each time.");
    diags.Report(diagID);
  }
  if (moduleDirList.size() == 1)
    res.setModuleDir(moduleDirList[0]);

  // -fdebug-module-writer option
  if (args.hasArg(clang::driver::options::OPT_fdebug_module_writer)) {
    res.setDebugModuleDir(true);
  }

  // -fhermetic-module-files option
  if (args.hasArg(clang::driver::options::OPT_fhermetic_module_files)) {
    res.setHermeticModuleFileOutput(true);
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
/// FC1 driver entry point for parsing diagnostic arguments.
static bool parseDiagArgs(CompilerInvocation &res, llvm::opt::ArgList &args,
                          clang::DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  auto &features{res.getFrontendOpts().features};
  // The order of these flags (-pedantic -W<feature> -w) is important and is
  // chosen to match clang's behavior.

  // -pedantic
  if (args.hasArg(clang::driver::options::OPT_pedantic)) {
    features.WarnOnAllNonstandard();
    features.WarnOnAllUsage();
    res.setEnableConformanceChecks();
    res.setEnableUsageChecks();
  }

  // -Werror option
  // TODO: Currently throws a Diagnostic for anything other than -W<error>,
  // this has to change when other -W<opt>'s are supported.
  if (args.hasArg(clang::driver::options::OPT_W_Joined)) {
    const auto &wArgs =
        args.getAllArgValues(clang::driver::options::OPT_W_Joined);
    for (const auto &wArg : wArgs) {
      if (wArg == "error") {
        res.setWarnAsErr(true);
        // -Wfatal-errors
      } else if (wArg == "fatal-errors") {
        res.setMaxErrors(1);
        // -W[no-]<feature>
      } else if (!features.EnableWarning(wArg)) {
        const unsigned diagID = diags.getCustomDiagID(
            clang::DiagnosticsEngine::Error, "Unknown diagnostic option: -W%0");
        diags.Report(diagID) << wArg;
      }
    }
  }

  // -w
  if (args.hasArg(clang::driver::options::OPT_w)) {
    features.DisableAllWarnings();
    res.setDisableWarnings();
  }

  // Default to off for `flang -fc1`.
  bool showColors{parseShowColorsArgs(args, false)};
  diags.getDiagnosticOptions().ShowColors = showColors;
  res.getDiagnosticOpts().ShowColors = showColors;
  res.getFrontendOpts().showColors = showColors;
  return diags.getNumErrors() == numErrorsBefore;
}

/// Parses all Dialect related arguments and populates the variables
/// options accordingly. Returns false if new errors are generated.
static bool parseDialectArgs(CompilerInvocation &res, llvm::opt::ArgList &args,
                             clang::DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // -fd-lines-as-code
  if (args.hasArg(clang::driver::options::OPT_fd_lines_as_code)) {
    if (res.getFrontendOpts().fortranForm == FortranForm::FreeForm) {
      const unsigned fdLinesAsWarning = diags.getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "‘-fd-lines-as-code’ has no effect in free form.");
      diags.Report(fdLinesAsWarning);
    } else {
      res.getFrontendOpts().features.Enable(
          Fortran::common::LanguageFeature::OldDebugLines, true);
    }
  }

  // -fd-lines-as-comments
  if (args.hasArg(clang::driver::options::OPT_fd_lines_as_comments)) {
    if (res.getFrontendOpts().fortranForm == FortranForm::FreeForm) {
      const unsigned fdLinesAsWarning = diags.getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "‘-fd-lines-as-comments’ has no effect in free form.");
      diags.Report(fdLinesAsWarning);
    } else {
      res.getFrontendOpts().features.Enable(
          Fortran::common::LanguageFeature::OldDebugLines, false);
    }
  }

  // -fdefault* family
  if (args.hasArg(clang::driver::options::OPT_fdefault_real_8)) {
    res.getDefaultKinds().set_defaultRealKind(8);
    res.getDefaultKinds().set_doublePrecisionKind(16);
  }
  if (args.hasArg(clang::driver::options::OPT_fdefault_integer_8)) {
    res.getDefaultKinds().set_defaultIntegerKind(8);
    res.getDefaultKinds().set_subscriptIntegerKind(8);
    res.getDefaultKinds().set_sizeIntegerKind(8);
    res.getDefaultKinds().set_defaultLogicalKind(8);
  }
  if (args.hasArg(clang::driver::options::OPT_fdefault_double_8)) {
    if (!args.hasArg(clang::driver::options::OPT_fdefault_real_8)) {
      // -fdefault-double-8 has to be used with -fdefault-real-8
      // to be compatible with gfortran
      const unsigned diagID = diags.getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "Use of `-fdefault-double-8` requires `-fdefault-real-8`");
      diags.Report(diagID);
    }
    // https://gcc.gnu.org/onlinedocs/gfortran/Fortran-Dialect-Options.html
    res.getDefaultKinds().set_doublePrecisionKind(8);
  }
  if (args.hasArg(clang::driver::options::OPT_flarge_sizes))
    res.getDefaultKinds().set_sizeIntegerKind(8);

  // -x cuda
  auto language = args.getLastArgValue(clang::driver::options::OPT_x);
  if (language == "cuda") {
    res.getFrontendOpts().features.Enable(
        Fortran::common::LanguageFeature::CUDA);
  }

  // -fopenacc
  if (args.hasArg(clang::driver::options::OPT_fopenacc)) {
    res.getFrontendOpts().features.Enable(
        Fortran::common::LanguageFeature::OpenACC);
  }

  // -std=f2018
  // TODO: Set proper options when more fortran standards
  // are supported.
  if (args.hasArg(clang::driver::options::OPT_std_EQ)) {
    auto standard = args.getLastArgValue(clang::driver::options::OPT_std_EQ);
    // We only allow f2018 as the given standard
    if (standard == "f2018") {
      res.setEnableConformanceChecks();
      res.getFrontendOpts().features.WarnOnAllNonstandard();
    } else {
      const unsigned diagID =
          diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                "Only -std=f2018 is allowed currently.");
      diags.Report(diagID);
    }
  }
  // -fcoarray
  if (args.hasArg(clang::driver::options::OPT_fcoarray)) {
    res.getFrontendOpts().features.Enable(
        Fortran::common::LanguageFeature::Coarray);
    const unsigned diagID =
        diags.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                              "Support for multi image Fortran features is "
                              "still experimental and in development.");
    diags.Report(diagID);
  }

  return diags.getNumErrors() == numErrorsBefore;
}

/// Parses all OpenMP related arguments if the -fopenmp option is present,
/// populating the \c res object accordingly. Returns false if new errors are
/// generated.
static bool parseOpenMPArgs(CompilerInvocation &res, llvm::opt::ArgList &args,
                            clang::DiagnosticsEngine &diags) {
  llvm::opt::Arg *arg = args.getLastArg(clang::driver::options::OPT_fopenmp,
                                        clang::driver::options::OPT_fno_openmp);
  if (!arg ||
      arg->getOption().matches(clang::driver::options::OPT_fno_openmp)) {
    bool isSimdSpecified = args.hasFlag(
        clang::driver::options::OPT_fopenmp_simd,
        clang::driver::options::OPT_fno_openmp_simd, /*Default=*/false);
    if (!isSimdSpecified)
      return true;
    res.getLangOpts().OpenMPSimd = 1;
  }

  unsigned numErrorsBefore = diags.getNumErrors();
  llvm::Triple t(res.getTargetOpts().triple);

  constexpr unsigned newestFullySupported = 31;
  constexpr unsigned latestFinalized = 60;
  // By default OpenMP is set to the most recent fully supported version
  res.getLangOpts().OpenMPVersion = newestFullySupported;
  res.getFrontendOpts().features.Enable(
      Fortran::common::LanguageFeature::OpenMP);
  if (auto *arg =
          args.getLastArg(clang::driver::options::OPT_fopenmp_version_EQ)) {
    llvm::ArrayRef<unsigned> ompVersions = llvm::omp::getOpenMPVersions();
    unsigned oldVersions[] = {11, 20, 25, 30};
    unsigned version = 0;

    auto reportBadVersion = [&](llvm::StringRef value) {
      const unsigned diagID =
          diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                "'%0' is not a valid OpenMP version in '%1', "
                                "valid versions are %2");
      std::string buffer;
      llvm::raw_string_ostream versions(buffer);
      llvm::interleaveComma(ompVersions, versions);

      diags.Report(diagID) << value << arg->getAsString(args) << versions.str();
    };

    auto reportFutureVersion = [&](llvm::StringRef value) {
      const unsigned diagID = diags.getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "The specification for OpenMP version %0 is still under development; "
          "the syntax and semantics of new features may be subject to change");
      std::string buffer;
      llvm::raw_string_ostream versions(buffer);
      llvm::interleaveComma(ompVersions, versions);

      diags.Report(diagID) << value;
    };

    llvm::StringRef value = arg->getValue();
    if (!value.getAsInteger(/*radix=*/10, version)) {
      if (llvm::is_contained(ompVersions, version)) {
        res.getLangOpts().OpenMPVersion = version;

        if (version > latestFinalized)
          reportFutureVersion(value);
        else if (version > newestFullySupported)
          diags.Report(clang::diag::warn_openmp_incomplete) << version;
      } else if (llvm::is_contained(oldVersions, version)) {
        const unsigned diagID =
            diags.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                  "OpenMP version %0 is no longer supported, "
                                  "assuming version %1");
        std::string assumed = std::to_string(res.getLangOpts().OpenMPVersion);
        diags.Report(diagID) << value << assumed;
      } else {
        reportBadVersion(value);
      }
    } else {
      reportBadVersion(value);
    }
  }

  if (args.hasArg(clang::driver::options::OPT_fopenmp_force_usm)) {
    res.getLangOpts().OpenMPForceUSM = 1;
  }
  if (args.hasArg(clang::driver::options::OPT_fopenmp_is_target_device)) {
    res.getLangOpts().OpenMPIsTargetDevice = 1;

    // Get OpenMP host file path if any and report if a non existent file is
    // found
    if (auto *arg = args.getLastArg(
            clang::driver::options::OPT_fopenmp_host_ir_file_path)) {
      res.getLangOpts().OMPHostIRFile = arg->getValue();
      if (!llvm::sys::fs::exists(res.getLangOpts().OMPHostIRFile))
        diags.Report(clang::diag::err_drv_omp_host_ir_file_not_found)
            << res.getLangOpts().OMPHostIRFile;
    }

    if (args.hasFlag(
            clang::driver::options::OPT_fopenmp_assume_teams_oversubscription,
            clang::driver::options::
                OPT_fno_openmp_assume_teams_oversubscription,
            /*Default=*/false))
      res.getLangOpts().OpenMPTeamSubscription = true;

    if (args.hasArg(clang::driver::options::OPT_fopenmp_assume_no_thread_state))
      res.getLangOpts().OpenMPNoThreadState = 1;

    if (args.hasArg(
            clang::driver::options::OPT_fopenmp_assume_no_nested_parallelism))
      res.getLangOpts().OpenMPNoNestedParallelism = 1;

    if (args.hasFlag(
            clang::driver::options::OPT_fopenmp_assume_threads_oversubscription,
            clang::driver::options::
                OPT_fno_openmp_assume_threads_oversubscription,
            /*Default=*/false))
      res.getLangOpts().OpenMPThreadSubscription = true;

    if ((args.hasArg(clang::driver::options::OPT_fopenmp_target_debug) ||
         args.hasArg(clang::driver::options::OPT_fopenmp_target_debug_EQ))) {
      res.getLangOpts().OpenMPTargetDebug = getLastArgIntValue(
          args, clang::driver::options::OPT_fopenmp_target_debug_EQ,
          res.getLangOpts().OpenMPTargetDebug, diags);

      if (!res.getLangOpts().OpenMPTargetDebug &&
          args.hasArg(clang::driver::options::OPT_fopenmp_target_debug))
        res.getLangOpts().OpenMPTargetDebug = 1;
    }
    if (args.hasArg(clang::driver::options::OPT_no_offloadlib))
      res.getLangOpts().NoGPULib = 1;
  }
  if (llvm::Triple(res.getTargetOpts().triple).isGPU()) {
    if (!res.getLangOpts().OpenMPIsTargetDevice) {
      const unsigned diagID = diags.getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "OpenMP GPU is only prepared to deal with device code.");
      diags.Report(diagID);
    }
    res.getLangOpts().OpenMPIsGPU = 1;
  } else {
    res.getLangOpts().OpenMPIsGPU = 0;
  }

  // Get the OpenMP target triples if any.
  if (auto *arg =
          args.getLastArg(clang::driver::options::OPT_offload_targets_EQ)) {
    enum ArchPtrSize { Arch16Bit, Arch32Bit, Arch64Bit };
    auto getArchPtrSize = [](const llvm::Triple &triple) {
      if (triple.isArch16Bit())
        return Arch16Bit;
      if (triple.isArch32Bit())
        return Arch32Bit;
      assert(triple.isArch64Bit() && "Expected 64-bit architecture");
      return Arch64Bit;
    };

    for (unsigned i = 0; i < arg->getNumValues(); ++i) {
      llvm::Triple tt(arg->getValue(i));

      if (tt.getArch() == llvm::Triple::UnknownArch ||
          !(tt.getArch() == llvm::Triple::aarch64 || tt.isPPC() ||
            tt.getArch() == llvm::Triple::systemz ||
            tt.getArch() == llvm::Triple::x86 ||
            tt.getArch() == llvm::Triple::x86_64 || tt.isGPU()))
        diags.Report(clang::diag::err_drv_invalid_omp_target)
            << arg->getValue(i);
      else if (getArchPtrSize(t) != getArchPtrSize(tt))
        diags.Report(clang::diag::err_drv_incompatible_omp_arch)
            << arg->getValue(i) << t.str();
      else
        res.getLangOpts().OMPTargetTriples.push_back(tt);
    }
  }
  return diags.getNumErrors() == numErrorsBefore;
}

/// Parses signed integer overflow options and populates the
/// CompilerInvocation accordingly.
/// Returns false if new errors are generated.
///
/// \param [out] invoc Stores the processed arguments
/// \param [in] args The compiler invocation arguments to parse
/// \param [out] diags DiagnosticsEngine to report erros with
static bool parseIntegerOverflowArgs(CompilerInvocation &invoc,
                                     llvm::opt::ArgList &args,
                                     clang::DiagnosticsEngine &diags) {
  Fortran::common::LangOptions &opts = invoc.getLangOpts();

  if (args.getLastArg(clang::driver::options::OPT_fwrapv))
    opts.setSignedOverflowBehavior(Fortran::common::LangOptions::SOB_Defined);

  return true;
}

/// Parses all floating point related arguments and populates the
/// CompilerInvocation accordingly.
/// Returns false if new errors are generated.
///
/// \param [out] invoc Stores the processed arguments
/// \param [in] args The compiler invocation arguments to parse
/// \param [out] diags DiagnosticsEngine to report erros with
static bool parseFloatingPointArgs(CompilerInvocation &invoc,
                                   llvm::opt::ArgList &args,
                                   clang::DiagnosticsEngine &diags) {
  Fortran::common::LangOptions &opts = invoc.getLangOpts();

  if (const llvm::opt::Arg *a =
          args.getLastArg(clang::driver::options::OPT_ffp_contract)) {
    const llvm::StringRef val = a->getValue();
    enum Fortran::common::LangOptions::FPModeKind fpContractMode;

    if (val == "off")
      fpContractMode = Fortran::common::LangOptions::FPM_Off;
    else if (val == "fast")
      fpContractMode = Fortran::common::LangOptions::FPM_Fast;
    else {
      diags.Report(clang::diag::err_drv_unsupported_option_argument)
          << a->getSpelling() << val;
      return false;
    }

    opts.setFPContractMode(fpContractMode);
  }

  if (args.getLastArg(clang::driver::options::OPT_menable_no_infs)) {
    opts.NoHonorInfs = true;
  }

  if (args.getLastArg(clang::driver::options::OPT_menable_no_nans)) {
    opts.NoHonorNaNs = true;
  }

  if (args.getLastArg(clang::driver::options::OPT_fapprox_func)) {
    opts.ApproxFunc = true;
  }

  if (args.getLastArg(clang::driver::options::OPT_fno_signed_zeros)) {
    opts.NoSignedZeros = true;
  }

  if (args.getLastArg(clang::driver::options::OPT_mreassociate)) {
    opts.AssociativeMath = true;
  }

  if (args.getLastArg(clang::driver::options::OPT_freciprocal_math)) {
    opts.ReciprocalMath = true;
  }

  if (args.getLastArg(clang::driver::options::OPT_ffast_math)) {
    opts.NoHonorInfs = true;
    opts.NoHonorNaNs = true;
    opts.AssociativeMath = true;
    opts.ReciprocalMath = true;
    opts.ApproxFunc = true;
    opts.NoSignedZeros = true;
    opts.setFPContractMode(Fortran::common::LangOptions::FPM_Fast);
  }

  return true;
}

/// Parses vscale range options and populates the CompilerInvocation
/// accordingly.
/// Returns false if new errors are generated.
///
/// \param [out] invoc Stores the processed arguments
/// \param [in] args The compiler invocation arguments to parse
/// \param [out] diags DiagnosticsEngine to report erros with
static bool parseVScaleArgs(CompilerInvocation &invoc, llvm::opt::ArgList &args,
                            clang::DiagnosticsEngine &diags) {
  const auto *vscaleMin =
      args.getLastArg(clang::driver::options::OPT_mvscale_min_EQ);
  const auto *vscaleMax =
      args.getLastArg(clang::driver::options::OPT_mvscale_max_EQ);

  if (!vscaleMin && !vscaleMax)
    return true;

  llvm::Triple triple = llvm::Triple(invoc.getTargetOpts().triple);
  if (!triple.isAArch64() && !triple.isRISCV()) {
    const unsigned diagID =
        diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                              "`-mvscale-max` and `-mvscale-min` are not "
                              "supported for this architecture: %0");
    diags.Report(diagID) << triple.getArchName();
    return false;
  }

  Fortran::common::LangOptions &opts = invoc.getLangOpts();
  if (vscaleMin) {
    llvm::StringRef argValue = llvm::StringRef(vscaleMin->getValue());
    unsigned vscaleMinVal;
    if (argValue.getAsInteger(/*Radix=*/10, vscaleMinVal)) {
      diags.Report(clang::diag::err_drv_unsupported_option_argument)
          << vscaleMax->getSpelling() << argValue;
      return false;
    }
    opts.VScaleMin = vscaleMinVal;
  }

  if (vscaleMax) {
    llvm::StringRef argValue = llvm::StringRef(vscaleMax->getValue());
    unsigned vscaleMaxVal;
    if (argValue.getAsInteger(/*Radix=w*/ 10, vscaleMaxVal)) {
      diags.Report(clang::diag::err_drv_unsupported_option_argument)
          << vscaleMax->getSpelling() << argValue;
      return false;
    }
    opts.VScaleMax = vscaleMaxVal;
  }
  return true;
}

static bool parseLinkerOptionsArgs(CompilerInvocation &invoc,
                                   llvm::opt::ArgList &args,
                                   clang::DiagnosticsEngine &diags) {
  llvm::Triple triple = llvm::Triple(invoc.getTargetOpts().triple);

  // TODO: support --dependent-lib on other platforms when MLIR supports
  //       !llvm.dependent.lib
  if (args.hasArg(clang::driver::options::OPT_dependent_lib) &&
      !triple.isOSWindows()) {
    const unsigned diagID =
        diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                              "--dependent-lib is only supported on Windows");
    diags.Report(diagID);
    return false;
  }

  invoc.getCodeGenOpts().DependentLibs =
      args.getAllArgValues(clang::driver::options::OPT_dependent_lib);
  return true;
}

static bool parseLangOptionsArgs(CompilerInvocation &invoc,
                                 llvm::opt::ArgList &args,
                                 clang::DiagnosticsEngine &diags) {
  bool success = true;

  success &= parseIntegerOverflowArgs(invoc, args, diags);
  success &= parseFloatingPointArgs(invoc, args, diags);
  success &= parseVScaleArgs(invoc, args, diags);

  return success;
}

bool CompilerInvocation::createFromArgs(
    CompilerInvocation &invoc, llvm::ArrayRef<const char *> commandLineArgs,
    clang::DiagnosticsEngine &diags, const char *argv0) {

  bool success = true;

  // Set the default triple for this CompilerInvocation. This might be
  // overridden by users with `-triple` (see the call to `ParseTargetArgs`
  // below).
  // NOTE: Like in Clang, it would be nice to use option marshalling
  // for this so that the entire logic for setting-up the triple is in one
  // place.
  invoc.getTargetOpts().triple =
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple());

  // Parse the arguments
  const llvm::opt::OptTable &opts = clang::driver::getDriverOptTable();
  llvm::opt::Visibility visibilityMask(clang::driver::options::FC1Option);
  unsigned missingArgIndex, missingArgCount;
  llvm::opt::InputArgList args = opts.ParseArgs(
      commandLineArgs, missingArgIndex, missingArgCount, visibilityMask);

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
    if (opts.findNearest(argString, nearest, visibilityMask) > 1)
      diags.Report(clang::diag::err_drv_unknown_argument) << argString;
    else
      diags.Report(clang::diag::err_drv_unknown_argument_with_suggestion)
          << argString << nearest;
    success = false;
  }

  // -flang-experimental-hlfir
  if (args.hasArg(clang::driver::options::OPT_flang_experimental_hlfir) ||
      args.hasArg(clang::driver::options::OPT_emit_hlfir)) {
    invoc.loweringOpts.setLowerToHighLevelFIR(true);
  }

  // -flang-deprecated-no-hlfir
  if (args.hasArg(clang::driver::options::OPT_flang_deprecated_no_hlfir) &&
      !args.hasArg(clang::driver::options::OPT_emit_hlfir)) {
    if (args.hasArg(clang::driver::options::OPT_flang_experimental_hlfir)) {
      const unsigned diagID = diags.getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "Options '-flang-experimental-hlfir' and "
          "'-flang-deprecated-no-hlfir' cannot be both specified");
      diags.Report(diagID);
    }
    invoc.loweringOpts.setLowerToHighLevelFIR(false);
  }

  // -fno-ppc-native-vector-element-order
  if (args.hasArg(clang::driver::options::OPT_fno_ppc_native_vec_elem_order)) {
    invoc.loweringOpts.setNoPPCNativeVecElemOrder(true);
  }

  // -f[no-]init-global-zero
  if (args.hasFlag(clang::driver::options::OPT_finit_global_zero,
                   clang::driver::options::OPT_fno_init_global_zero,
                   /*default=*/true))
    invoc.loweringOpts.setInitGlobalZero(true);
  else
    invoc.loweringOpts.setInitGlobalZero(false);

  // Preserve all the remark options requested, i.e. -Rpass, -Rpass-missed or
  // -Rpass-analysis. This will be used later when processing and outputting the
  // remarks generated by LLVM in ExecuteCompilerInvocation.cpp.
  for (auto *a : args.filtered(clang::driver::options::OPT_R_Group)) {
    if (a->getOption().matches(clang::driver::options::OPT_R_value_Group))
      // This is -Rfoo=, where foo is the name of the diagnostic
      // group. Add only the remark option name to the diagnostics. e.g. for
      // -Rpass= we will add the string "pass".
      invoc.getDiagnosticOpts().Remarks.push_back(
          std::string(a->getOption().getName().drop_front(1).rtrim("=-")));
    else
      // If no regex was provided, add the provided value, e.g. for -Rpass add
      // the string "pass".
      invoc.getDiagnosticOpts().Remarks.push_back(a->getValue());
  }

  // -frealloc-lhs is the default.
  if (!args.hasFlag(clang::driver::options::OPT_frealloc_lhs,
                    clang::driver::options::OPT_fno_realloc_lhs, true))
    invoc.loweringOpts.setReallocateLHS(false);

  invoc.loweringOpts.setRepackArrays(
      args.hasFlag(clang::driver::options::OPT_frepack_arrays,
                   clang::driver::options::OPT_fno_repack_arrays,
                   /*default=*/false));
  invoc.loweringOpts.setStackRepackArrays(
      args.hasFlag(clang::driver::options::OPT_fstack_repack_arrays,
                   clang::driver::options::OPT_fno_stack_repack_arrays,
                   /*default=*/false));
  if (auto *arg = args.getLastArg(
          clang::driver::options::OPT_frepack_arrays_contiguity_EQ))
    invoc.loweringOpts.setRepackArraysWhole(arg->getValue() ==
                                            llvm::StringRef{"whole"});

  success &= parseFrontendArgs(invoc.getFrontendOpts(), args, diags);
  parseTargetArgs(invoc.getTargetOpts(), args);
  parsePreprocessorArgs(invoc.getPreprocessorOpts(), args);
  parseCodeGenArgs(invoc.getCodeGenOpts(), args, diags);
  success &= parseDebugArgs(invoc.getCodeGenOpts(), args, diags);
  success &= parseVectorLibArg(invoc.getCodeGenOpts(), args, diags);
  success &= parseSemaArgs(invoc, args, diags);
  success &= parseDialectArgs(invoc, args, diags);
  success &= parseOpenMPArgs(invoc, args, diags);
  success &= parseDiagArgs(invoc, args, diags);

  // Collect LLVM (-mllvm) and MLIR (-mmlir) options.
  // NOTE: Try to avoid adding any options directly to `llvmArgs` or
  // `mlirArgs`. Instead, you can use
  //    * `-mllvm <your-llvm-option>`, or
  //    * `-mmlir <your-mlir-option>`.
  invoc.frontendOpts.llvmArgs =
      args.getAllArgValues(clang::driver::options::OPT_mllvm);
  invoc.frontendOpts.mlirArgs =
      args.getAllArgValues(clang::driver::options::OPT_mmlir);

  success &= parseLangOptionsArgs(invoc, args, diags);

  success &= parseLinkerOptionsArgs(invoc, args, diags);

  // Set the string to be used as the return value of the COMPILER_OPTIONS
  // intrinsic of iso_fortran_env. This is either passed in from the parent
  // compiler driver invocation with an environment variable, or failing that
  // set to the command line arguments of the frontend driver invocation.
  invoc.allCompilerInvocOpts = std::string();
  llvm::raw_string_ostream os(invoc.allCompilerInvocOpts);
  char *compilerOptsEnv = std::getenv("FLANG_COMPILER_OPTIONS_STRING");
  if (compilerOptsEnv != nullptr) {
    os << compilerOptsEnv;
  } else {
    os << argv0 << ' ';
    for (auto it = commandLineArgs.begin(), e = commandLineArgs.end(); it != e;
         ++it) {
      os << ' ' << *it;
    }
  }

  // Process the timing-related options.
  if (args.hasArg(clang::driver::options::OPT_ftime_report))
    invoc.enableTimers = true;

  invoc.setArgv0(argv0);

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
  fortranOptions.searchDirectories.emplace_back(
      getOpenMPHeadersDir(getArgv0()));

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
  fortranOptions.predefinitions.emplace_back("__flang_major__",
                                             FLANG_VERSION_MAJOR_STRING);
  fortranOptions.predefinitions.emplace_back("__flang_minor__",
                                             FLANG_VERSION_MINOR_STRING);
  fortranOptions.predefinitions.emplace_back("__flang_patchlevel__",
                                             FLANG_VERSION_PATCHLEVEL_STRING);

  // Add predefinitions based on the relocation model
  if (unsigned PICLevel = getCodeGenOpts().PICLevel) {
    fortranOptions.predefinitions.emplace_back("__PIC__",
                                               std::to_string(PICLevel));
    fortranOptions.predefinitions.emplace_back("__pic__",
                                               std::to_string(PICLevel));
    if (getCodeGenOpts().IsPIE) {
      fortranOptions.predefinitions.emplace_back("__PIE__",
                                                 std::to_string(PICLevel));
      fortranOptions.predefinitions.emplace_back("__pie__",
                                                 std::to_string(PICLevel));
    }
  }

  // Add predefinitions based on extensions enabled
  if (frontendOptions.features.IsEnabled(
          Fortran::common::LanguageFeature::OpenACC)) {
    fortranOptions.predefinitions.emplace_back("_OPENACC", "202211");
  }
  if (frontendOptions.features.IsEnabled(
          Fortran::common::LanguageFeature::OpenMP)) {
    Fortran::common::setOpenMPMacro(getLangOpts().OpenMPVersion,
                                    fortranOptions.predefinitions);
  }

  llvm::Triple targetTriple{llvm::Triple(this->targetOpts.triple)};
  if (targetTriple.isOSLinux()) {
    fortranOptions.predefinitions.emplace_back("__linux__", "1");
  } else if (targetTriple.isOSAIX()) {
    fortranOptions.predefinitions.emplace_back("_AIX", "1");
  }

  switch (targetTriple.getArch()) {
  default:
    break;
  case llvm::Triple::ArchType::x86_64:
    fortranOptions.predefinitions.emplace_back("__x86_64__", "1");
    fortranOptions.predefinitions.emplace_back("__x86_64", "1");
    break;
  case llvm::Triple::ArchType::ppc:
  case llvm::Triple::ArchType::ppc64:
  case llvm::Triple::ArchType::ppcle:
  case llvm::Triple::ArchType::ppc64le:
    // '__powerpc__' is a generic macro for any PowerPC.
    fortranOptions.predefinitions.emplace_back("__powerpc__", "1");
    if (targetTriple.isOSAIX() && targetTriple.isArch64Bit()) {
      fortranOptions.predefinitions.emplace_back("__64BIT__", "1");
    }
    break;
  case llvm::Triple::ArchType::aarch64:
    fortranOptions.predefinitions.emplace_back("__aarch64__", "1");
    fortranOptions.predefinitions.emplace_back("__aarch64", "1");
    break;
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

  // -E
  fortranOptions.prescanAndReformat =
      frontendOptions.programAction == PrintPreprocessedInput;

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
  fortranOptions.intrinsicModuleDirectories.emplace_back(
      getIntrinsicDir(getArgv0()));

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

  fortranOptions.features = frontendOptions.features;
}

std::unique_ptr<Fortran::semantics::SemanticsContext>
CompilerInvocation::getSemanticsCtx(
    Fortran::parser::AllCookedSources &allCookedSources,
    const llvm::TargetMachine &targetMachine) {
  auto &fortranOptions = getFortranOpts();

  auto semanticsContext = std::make_unique<semantics::SemanticsContext>(
      getDefaultKinds(), fortranOptions.features, getLangOpts(),
      allCookedSources);

  semanticsContext->set_moduleDirectory(getModuleDir())
      .set_searchDirectories(fortranOptions.searchDirectories)
      .set_intrinsicModuleDirectories(fortranOptions.intrinsicModuleDirectories)
      .set_maxErrors(getMaxErrors())
      .set_warningsAreErrors(getWarnAsErr())
      .set_moduleFileSuffix(getModuleFileSuffix())
      .set_underscoring(getCodeGenOpts().Underscoring);

  std::string compilerVersion = Fortran::common::getFlangFullVersion();
  Fortran::tools::setUpTargetCharacteristics(
      semanticsContext->targetCharacteristics(), targetMachine, getTargetOpts(),
      compilerVersion, allCompilerInvocOpts);
  return semanticsContext;
}

/// Set \p loweringOptions controlling lowering behavior based
/// on the \p optimizationLevel.
void CompilerInvocation::setLoweringOptions() {
  const CodeGenOptions &codegenOpts = getCodeGenOpts();

  // Lower TRANSPOSE as a runtime call under -O0.
  loweringOpts.setOptimizeTranspose(codegenOpts.OptimizationLevel > 0);
  loweringOpts.setUnderscoring(codegenOpts.Underscoring);
  loweringOpts.setSkipExternalRttiDefinition(skipExternalRttiDefinition);

  const Fortran::common::LangOptions &langOptions = getLangOpts();
  loweringOpts.setIntegerWrapAround(langOptions.getSignedOverflowBehavior() ==
                                    Fortran::common::LangOptions::SOB_Defined);
  Fortran::common::MathOptionsBase &mathOpts = loweringOpts.getMathOptions();
  // TODO: when LangOptions are finalized, we can represent
  //       the math related options using Fortran::commmon::MathOptionsBase,
  //       so that we can just copy it into LoweringOptions.
  mathOpts
      .setFPContractEnabled(langOptions.getFPContractMode() ==
                            Fortran::common::LangOptions::FPM_Fast)
      .setNoHonorInfs(langOptions.NoHonorInfs)
      .setNoHonorNaNs(langOptions.NoHonorNaNs)
      .setApproxFunc(langOptions.ApproxFunc)
      .setNoSignedZeros(langOptions.NoSignedZeros)
      .setAssociativeMath(langOptions.AssociativeMath)
      .setReciprocalMath(langOptions.ReciprocalMath);

  if (codegenOpts.getComplexRange() ==
          CodeGenOptions::ComplexRangeKind::CX_Improved ||
      codegenOpts.getComplexRange() ==
          CodeGenOptions::ComplexRangeKind::CX_Basic)
    loweringOpts.setComplexDivisionToRuntime(false);
}
