//===--- CompilerInvocation.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestModuleFileExtension.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Util.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/LangStandard.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Serialization/ModuleFileExtension.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Target/TargetOptions.h"
#include <atomic>
#include <memory>
#include <sys/stat.h>
#include <system_error>
using namespace clang;

//===----------------------------------------------------------------------===//
// Initialization.
//===----------------------------------------------------------------------===//

CompilerInvocationBase::CompilerInvocationBase()
  : LangOpts(new LangOptions()), TargetOpts(new TargetOptions()),
    DiagnosticOpts(new DiagnosticOptions()),
    HeaderSearchOpts(new HeaderSearchOptions()),
    PreprocessorOpts(new PreprocessorOptions()) {}

CompilerInvocationBase::CompilerInvocationBase(const CompilerInvocationBase &X)
  : RefCountedBase<CompilerInvocation>(),
    LangOpts(new LangOptions(*X.getLangOpts())),
    TargetOpts(new TargetOptions(X.getTargetOpts())),
    DiagnosticOpts(new DiagnosticOptions(X.getDiagnosticOpts())),
    HeaderSearchOpts(new HeaderSearchOptions(X.getHeaderSearchOpts())),
    PreprocessorOpts(new PreprocessorOptions(X.getPreprocessorOpts())) {}

CompilerInvocationBase::~CompilerInvocationBase() {}

//===----------------------------------------------------------------------===//
// Deserialization (from args)
//===----------------------------------------------------------------------===//

using namespace clang::driver;
using namespace clang::driver::options;
using namespace llvm::opt;

//

static unsigned getOptimizationLevel(ArgList &Args, InputKind IK,
                                     DiagnosticsEngine &Diags) {
  unsigned DefaultOpt = 0;
  if (IK == IK_OpenCL && !Args.hasArg(OPT_cl_opt_disable))
    DefaultOpt = 2;

  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().matches(options::OPT_O0))
      return 0;

    if (A->getOption().matches(options::OPT_Ofast))
      return 3;

    assert (A->getOption().matches(options::OPT_O));

    StringRef S(A->getValue());
    if (S == "s" || S == "z" || S.empty())
      return 2;

    return getLastArgIntValue(Args, OPT_O, DefaultOpt, Diags);
  }

  return DefaultOpt;
}

static unsigned getOptimizationLevelSize(ArgList &Args) {
  if (Arg *A = Args.getLastArg(options::OPT_O_Group)) {
    if (A->getOption().matches(options::OPT_O)) {
      switch (A->getValue()[0]) {
      default:
        return 0;
      case 's':
        return 1;
      case 'z':
        return 2;
      }
    }
  }
  return 0;
}

static void addDiagnosticArgs(ArgList &Args, OptSpecifier Group,
                              OptSpecifier GroupWithValue,
                              std::vector<std::string> &Diagnostics) {
  for (Arg *A : Args.filtered(Group)) {
    if (A->getOption().getKind() == Option::FlagClass) {
      // The argument is a pure flag (such as OPT_Wall or OPT_Wdeprecated). Add
      // its name (minus the "W" or "R" at the beginning) to the warning list.
      Diagnostics.push_back(A->getOption().getName().drop_front(1));
    } else if (A->getOption().matches(GroupWithValue)) {
      // This is -Wfoo= or -Rfoo=, where foo is the name of the diagnostic group.
      Diagnostics.push_back(A->getOption().getName().drop_front(1).rtrim("=-"));
    } else {
      // Otherwise, add its value (for OPT_W_Joined and similar).
      for (const char *Arg : A->getValues())
        Diagnostics.emplace_back(Arg);
    }
  }
}

static void getAllNoBuiltinFuncValues(ArgList &Args,
                                      std::vector<std::string> &Funcs) {
  SmallVector<const char *, 8> Values;
  for (const auto &Arg : Args) {
    const Option &O = Arg->getOption();
    if (O.matches(options::OPT_fno_builtin_)) {
      const char *FuncName = Arg->getValue();
      if (Builtin::Context::isBuiltinFunc(FuncName))
        Values.push_back(FuncName);
    }
  }
  Funcs.insert(Funcs.end(), Values.begin(), Values.end());
}

static bool ParseAnalyzerArgs(AnalyzerOptions &Opts, ArgList &Args,
                              DiagnosticsEngine &Diags) {
  using namespace options;
  bool Success = true;
  if (Arg *A = Args.getLastArg(OPT_analyzer_store)) {
    StringRef Name = A->getValue();
    AnalysisStores Value = llvm::StringSwitch<AnalysisStores>(Name)
#define ANALYSIS_STORE(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, NAME##Model)
#include "clang/StaticAnalyzer/Core/Analyses.def"
      .Default(NumStores);
    if (Value == NumStores) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.AnalysisStoreOpt = Value;
    }
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_constraints)) {
    StringRef Name = A->getValue();
    AnalysisConstraints Value = llvm::StringSwitch<AnalysisConstraints>(Name)
#define ANALYSIS_CONSTRAINTS(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, NAME##Model)
#include "clang/StaticAnalyzer/Core/Analyses.def"
      .Default(NumConstraints);
    if (Value == NumConstraints) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.AnalysisConstraintsOpt = Value;
    }
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_output)) {
    StringRef Name = A->getValue();
    AnalysisDiagClients Value = llvm::StringSwitch<AnalysisDiagClients>(Name)
#define ANALYSIS_DIAGNOSTICS(NAME, CMDFLAG, DESC, CREATFN) \
      .Case(CMDFLAG, PD_##NAME)
#include "clang/StaticAnalyzer/Core/Analyses.def"
      .Default(NUM_ANALYSIS_DIAG_CLIENTS);
    if (Value == NUM_ANALYSIS_DIAG_CLIENTS) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.AnalysisDiagOpt = Value;
    }
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_purge)) {
    StringRef Name = A->getValue();
    AnalysisPurgeMode Value = llvm::StringSwitch<AnalysisPurgeMode>(Name)
#define ANALYSIS_PURGE(NAME, CMDFLAG, DESC) \
      .Case(CMDFLAG, NAME)
#include "clang/StaticAnalyzer/Core/Analyses.def"
      .Default(NumPurgeModes);
    if (Value == NumPurgeModes) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.AnalysisPurgeOpt = Value;
    }
  }

  if (Arg *A = Args.getLastArg(OPT_analyzer_inlining_mode)) {
    StringRef Name = A->getValue();
    AnalysisInliningMode Value = llvm::StringSwitch<AnalysisInliningMode>(Name)
#define ANALYSIS_INLINING_MODE(NAME, CMDFLAG, DESC) \
      .Case(CMDFLAG, NAME)
#include "clang/StaticAnalyzer/Core/Analyses.def"
      .Default(NumInliningModes);
    if (Value == NumInliningModes) {
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.InliningMode = Value;
    }
  }

  Opts.ShowCheckerHelp = Args.hasArg(OPT_analyzer_checker_help);
  Opts.DisableAllChecks = Args.hasArg(OPT_analyzer_disable_all_checks);

  Opts.visualizeExplodedGraphWithGraphViz =
    Args.hasArg(OPT_analyzer_viz_egraph_graphviz);
  Opts.visualizeExplodedGraphWithUbiGraph =
    Args.hasArg(OPT_analyzer_viz_egraph_ubigraph);
  Opts.NoRetryExhausted = Args.hasArg(OPT_analyzer_disable_retry_exhausted);
  Opts.AnalyzeAll = Args.hasArg(OPT_analyzer_opt_analyze_headers);
  Opts.AnalyzerDisplayProgress = Args.hasArg(OPT_analyzer_display_progress);
  Opts.AnalyzeNestedBlocks =
    Args.hasArg(OPT_analyzer_opt_analyze_nested_blocks);
  Opts.eagerlyAssumeBinOpBifurcation = Args.hasArg(OPT_analyzer_eagerly_assume);
  Opts.AnalyzeSpecificFunction = Args.getLastArgValue(OPT_analyze_function);
  Opts.UnoptimizedCFG = Args.hasArg(OPT_analysis_UnoptimizedCFG);
  Opts.TrimGraph = Args.hasArg(OPT_trim_egraph);
  Opts.maxBlockVisitOnPath =
      getLastArgIntValue(Args, OPT_analyzer_max_loop, 4, Diags);
  Opts.PrintStats = Args.hasArg(OPT_analyzer_stats);
  Opts.InlineMaxStackDepth =
      getLastArgIntValue(Args, OPT_analyzer_inline_max_stack_depth,
                         Opts.InlineMaxStackDepth, Diags);

  Opts.CheckersControlList.clear();
  for (const Arg *A :
       Args.filtered(OPT_analyzer_checker, OPT_analyzer_disable_checker)) {
    A->claim();
    bool enable = (A->getOption().getID() == OPT_analyzer_checker);
    // We can have a list of comma separated checker names, e.g:
    // '-analyzer-checker=cocoa,unix'
    StringRef checkerList = A->getValue();
    SmallVector<StringRef, 4> checkers;
    checkerList.split(checkers, ",");
    for (StringRef checker : checkers)
      Opts.CheckersControlList.emplace_back(checker, enable);
  }

  // Go through the analyzer configuration options.
  for (const Arg *A : Args.filtered(OPT_analyzer_config)) {
    A->claim();
    // We can have a list of comma separated config names, e.g:
    // '-analyzer-config key1=val1,key2=val2'
    StringRef configList = A->getValue();
    SmallVector<StringRef, 4> configVals;
    configList.split(configVals, ",");
    for (unsigned i = 0, e = configVals.size(); i != e; ++i) {
      StringRef key, val;
      std::tie(key, val) = configVals[i].split("=");
      if (val.empty()) {
        Diags.Report(SourceLocation(),
                     diag::err_analyzer_config_no_value) << configVals[i];
        Success = false;
        break;
      }
      if (val.find('=') != StringRef::npos) {
        Diags.Report(SourceLocation(),
                     diag::err_analyzer_config_multiple_values)
          << configVals[i];
        Success = false;
        break;
      }
      Opts.Config[key] = val;
    }
  }

  return Success;
}

static bool ParseMigratorArgs(MigratorOptions &Opts, ArgList &Args) {
  Opts.NoNSAllocReallocError = Args.hasArg(OPT_migrator_no_nsalloc_error);
  Opts.NoFinalizeRemoval = Args.hasArg(OPT_migrator_no_finalize_removal);
  return true;
}

static void ParseCommentArgs(CommentOptions &Opts, ArgList &Args) {
  Opts.BlockCommandNames = Args.getAllArgValues(OPT_fcomment_block_commands);
  Opts.ParseAllComments = Args.hasArg(OPT_fparse_all_comments);
}

static StringRef getCodeModel(ArgList &Args, DiagnosticsEngine &Diags) {
  if (Arg *A = Args.getLastArg(OPT_mcode_model)) {
    StringRef Value = A->getValue();
    if (Value == "small" || Value == "kernel" || Value == "medium" ||
        Value == "large")
      return Value;
    Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Value;
  }
  return "default";
}

/// \brief Create a new Regex instance out of the string value in \p RpassArg.
/// It returns a pointer to the newly generated Regex instance.
static std::shared_ptr<llvm::Regex>
GenerateOptimizationRemarkRegex(DiagnosticsEngine &Diags, ArgList &Args,
                                Arg *RpassArg) {
  StringRef Val = RpassArg->getValue();
  std::string RegexError;
  std::shared_ptr<llvm::Regex> Pattern = std::make_shared<llvm::Regex>(Val);
  if (!Pattern->isValid(RegexError)) {
    Diags.Report(diag::err_drv_optimization_remark_pattern)
        << RegexError << RpassArg->getAsString(Args);
    Pattern.reset();
  }
  return Pattern;
}

static bool parseDiagnosticLevelMask(StringRef FlagName,
                                     const std::vector<std::string> &Levels,
                                     DiagnosticsEngine *Diags,
                                     DiagnosticLevelMask &M) {
  bool Success = true;
  for (const auto &Level : Levels) {
    DiagnosticLevelMask const PM =
      llvm::StringSwitch<DiagnosticLevelMask>(Level)
        .Case("note",    DiagnosticLevelMask::Note)
        .Case("remark",  DiagnosticLevelMask::Remark)
        .Case("warning", DiagnosticLevelMask::Warning)
        .Case("error",   DiagnosticLevelMask::Error)
        .Default(DiagnosticLevelMask::None);
    if (PM == DiagnosticLevelMask::None) {
      Success = false;
      if (Diags)
        Diags->Report(diag::err_drv_invalid_value) << FlagName << Level;
    }
    M = M | PM;
  }
  return Success;
}

static void parseSanitizerKinds(StringRef FlagName,
                                const std::vector<std::string> &Sanitizers,
                                DiagnosticsEngine &Diags, SanitizerSet &S) {
  for (const auto &Sanitizer : Sanitizers) {
    SanitizerMask K = parseSanitizerValue(Sanitizer, /*AllowGroups=*/false);
    if (K == 0)
      Diags.Report(diag::err_drv_invalid_value) << FlagName << Sanitizer;
    else
      S.set(K, true);
  }
}

static bool ParseCodeGenArgs(CodeGenOptions &Opts, ArgList &Args, InputKind IK,
                             DiagnosticsEngine &Diags,
                             const TargetOptions &TargetOpts) {
  using namespace options;
  bool Success = true;
  llvm::Triple Triple = llvm::Triple(TargetOpts.Triple);

  unsigned OptimizationLevel = getOptimizationLevel(Args, IK, Diags);
  // TODO: This could be done in Driver
  unsigned MaxOptLevel = 3;
  if (OptimizationLevel > MaxOptLevel) {
    // If the optimization level is not supported, fall back on the default
    // optimization
    Diags.Report(diag::warn_drv_optimization_value)
        << Args.getLastArg(OPT_O)->getAsString(Args) << "-O" << MaxOptLevel;
    OptimizationLevel = MaxOptLevel;
  }
  Opts.OptimizationLevel = OptimizationLevel;

  // We must always run at least the always inlining pass.
  Opts.setInlining(
    (Opts.OptimizationLevel > 1) ? CodeGenOptions::NormalInlining
                                 : CodeGenOptions::OnlyAlwaysInlining);
  // -fno-inline-functions overrides OptimizationLevel > 1.
  Opts.NoInline = Args.hasArg(OPT_fno_inline);
  Opts.setInlining(Args.hasArg(OPT_fno_inline_functions) ?
                     CodeGenOptions::OnlyAlwaysInlining : Opts.getInlining());

  if (Arg *A = Args.getLastArg(OPT_fveclib)) {
    StringRef Name = A->getValue();
    if (Name == "Accelerate")
      Opts.setVecLib(CodeGenOptions::Accelerate);
    else if (Name == "none")
      Opts.setVecLib(CodeGenOptions::NoLibrary);
    else
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Name;
  }

  if (Arg *A = Args.getLastArg(OPT_debug_info_kind_EQ)) {
    unsigned Val =
        llvm::StringSwitch<unsigned>(A->getValue())
            .Case("line-tables-only", codegenoptions::DebugLineTablesOnly)
            .Case("limited", codegenoptions::LimitedDebugInfo)
            .Case("standalone", codegenoptions::FullDebugInfo)
            .Default(~0U);
    if (Val == ~0U)
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args)
                                                << A->getValue();
    else
      Opts.setDebugInfo(static_cast<codegenoptions::DebugInfoKind>(Val));
  }
  if (Arg *A = Args.getLastArg(OPT_debugger_tuning_EQ)) {
    unsigned Val = llvm::StringSwitch<unsigned>(A->getValue())
                       .Case("gdb", CodeGenOptions::DebuggerKindGDB)
                       .Case("lldb", CodeGenOptions::DebuggerKindLLDB)
                       .Case("sce", CodeGenOptions::DebuggerKindSCE)
                       .Default(~0U);
    if (Val == ~0U)
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args)
                                                << A->getValue();
    else
      Opts.setDebuggerTuning(static_cast<CodeGenOptions::DebuggerKind>(Val));
  }
  Opts.DwarfVersion = getLastArgIntValue(Args, OPT_dwarf_version_EQ, 0, Diags);
  Opts.DebugColumnInfo = Args.hasArg(OPT_dwarf_column_info);
  Opts.EmitCodeView = Args.hasArg(OPT_gcodeview);
  Opts.SplitDwarfFile = Args.getLastArgValue(OPT_split_dwarf_file);
  Opts.DebugTypeExtRefs = Args.hasArg(OPT_dwarf_ext_refs);
  Opts.DebugExplicitImport = Triple.isPS4CPU(); 

  for (const auto &Arg : Args.getAllArgValues(OPT_fdebug_prefix_map_EQ))
    Opts.DebugPrefixMap.insert(StringRef(Arg).split('='));

  if (const Arg *A =
          Args.getLastArg(OPT_emit_llvm_uselists, OPT_no_emit_llvm_uselists))
    Opts.EmitLLVMUseLists = A->getOption().getID() == OPT_emit_llvm_uselists;

  Opts.DisableLLVMOpts = Args.hasArg(OPT_disable_llvm_optzns);
  Opts.DisableLLVMPasses = Args.hasArg(OPT_disable_llvm_passes);
  Opts.DisableRedZone = Args.hasArg(OPT_disable_red_zone);
  Opts.ForbidGuardVariables = Args.hasArg(OPT_fforbid_guard_variables);
  Opts.UseRegisterSizedBitfieldAccess = Args.hasArg(
    OPT_fuse_register_sized_bitfield_access);
  Opts.RelaxedAliasing = Args.hasArg(OPT_relaxed_aliasing);
  Opts.StructPathTBAA = !Args.hasArg(OPT_no_struct_path_tbaa);
  Opts.DwarfDebugFlags = Args.getLastArgValue(OPT_dwarf_debug_flags);
  Opts.MergeAllConstants = !Args.hasArg(OPT_fno_merge_all_constants);
  Opts.NoCommon = Args.hasArg(OPT_fno_common);
  Opts.NoImplicitFloat = Args.hasArg(OPT_no_implicit_float);
  Opts.OptimizeSize = getOptimizationLevelSize(Args);
  Opts.SimplifyLibCalls = !(Args.hasArg(OPT_fno_builtin) ||
                            Args.hasArg(OPT_ffreestanding));
  if (Opts.SimplifyLibCalls)
    getAllNoBuiltinFuncValues(Args, Opts.NoBuiltinFuncs);
  Opts.UnrollLoops =
      Args.hasFlag(OPT_funroll_loops, OPT_fno_unroll_loops,
                   (Opts.OptimizationLevel > 1 && !Opts.OptimizeSize));
  Opts.RerollLoops = Args.hasArg(OPT_freroll_loops);

  Opts.DisableIntegratedAS = Args.hasArg(OPT_fno_integrated_as);
  Opts.Autolink = !Args.hasArg(OPT_fno_autolink);
  Opts.SampleProfileFile = Args.getLastArgValue(OPT_fprofile_sample_use_EQ);
  Opts.ProfileInstrGenerate = Args.hasArg(OPT_fprofile_instr_generate) ||
      Args.hasArg(OPT_fprofile_instr_generate_EQ);
  Opts.InstrProfileOutput = Args.getLastArgValue(OPT_fprofile_instr_generate_EQ);
  Opts.InstrProfileInput = Args.getLastArgValue(OPT_fprofile_instr_use_EQ);
  Opts.CoverageMapping =
      Args.hasFlag(OPT_fcoverage_mapping, OPT_fno_coverage_mapping, false);
  Opts.DumpCoverageMapping = Args.hasArg(OPT_dump_coverage_mapping);
  Opts.AsmVerbose = Args.hasArg(OPT_masm_verbose);
  Opts.ObjCAutoRefCountExceptions = Args.hasArg(OPT_fobjc_arc_exceptions);
  Opts.CXAAtExit = !Args.hasArg(OPT_fno_use_cxa_atexit);
  Opts.CXXCtorDtorAliases = Args.hasArg(OPT_mconstructor_aliases);
  Opts.CodeModel = getCodeModel(Args, Diags);
  Opts.DebugPass = Args.getLastArgValue(OPT_mdebug_pass);
  Opts.DisableFPElim =
      (Args.hasArg(OPT_mdisable_fp_elim) || Args.hasArg(OPT_pg));
  Opts.DisableFree = Args.hasArg(OPT_disable_free);
  Opts.DiscardValueNames = Args.hasArg(OPT_discard_value_names);
  Opts.DisableTailCalls = Args.hasArg(OPT_mdisable_tail_calls);
  Opts.FloatABI = Args.getLastArgValue(OPT_mfloat_abi);
  if (Arg *A = Args.getLastArg(OPT_meabi)) {
    StringRef Value = A->getValue();
    llvm::EABI EABIVersion = llvm::StringSwitch<llvm::EABI>(Value)
                                 .Case("default", llvm::EABI::Default)
                                 .Case("4", llvm::EABI::EABI4)
                                 .Case("5", llvm::EABI::EABI5)
                                 .Case("gnu", llvm::EABI::GNU)
                                 .Default(llvm::EABI::Unknown);
    if (EABIVersion == llvm::EABI::Unknown)
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args)
                                                << Value;
    else
      Opts.EABIVersion = Value;
  }
  Opts.LessPreciseFPMAD = Args.hasArg(OPT_cl_mad_enable);
  Opts.LimitFloatPrecision = Args.getLastArgValue(OPT_mlimit_float_precision);
  Opts.NoInfsFPMath = (Args.hasArg(OPT_menable_no_infinities) ||
                       Args.hasArg(OPT_cl_finite_math_only) ||
                       Args.hasArg(OPT_cl_fast_relaxed_math));
  Opts.NoNaNsFPMath = (Args.hasArg(OPT_menable_no_nans) ||
                       Args.hasArg(OPT_cl_unsafe_math_optimizations) ||
                       Args.hasArg(OPT_cl_finite_math_only) ||
                       Args.hasArg(OPT_cl_fast_relaxed_math));
  Opts.NoSignedZeros = Args.hasArg(OPT_fno_signed_zeros);
  Opts.ReciprocalMath = Args.hasArg(OPT_freciprocal_math);
  Opts.NoZeroInitializedInBSS = Args.hasArg(OPT_mno_zero_initialized_in_bss);
  Opts.BackendOptions = Args.getAllArgValues(OPT_backend_option);
  Opts.NumRegisterParameters = getLastArgIntValue(Args, OPT_mregparm, 0, Diags);
  Opts.NoExecStack = Args.hasArg(OPT_mno_exec_stack);
  Opts.FatalWarnings = Args.hasArg(OPT_massembler_fatal_warnings);
  Opts.EnableSegmentedStacks = Args.hasArg(OPT_split_stacks);
  Opts.RelaxAll = Args.hasArg(OPT_mrelax_all);
  Opts.IncrementalLinkerCompatible =
      Args.hasArg(OPT_mincremental_linker_compatible);
  Opts.OmitLeafFramePointer = Args.hasArg(OPT_momit_leaf_frame_pointer);
  Opts.SaveTempLabels = Args.hasArg(OPT_msave_temp_labels);
  Opts.NoDwarfDirectoryAsm = Args.hasArg(OPT_fno_dwarf_directory_asm);
  Opts.SoftFloat = Args.hasArg(OPT_msoft_float);
  Opts.StrictEnums = Args.hasArg(OPT_fstrict_enums);
  Opts.StrictVTablePointers = Args.hasArg(OPT_fstrict_vtable_pointers);
  Opts.UnsafeFPMath = Args.hasArg(OPT_menable_unsafe_fp_math) ||
                      Args.hasArg(OPT_cl_unsafe_math_optimizations) ||
                      Args.hasArg(OPT_cl_fast_relaxed_math);
  Opts.UnwindTables = Args.hasArg(OPT_munwind_tables);
  Opts.RelocationModel = Args.getLastArgValue(OPT_mrelocation_model, "pic");
  Opts.ThreadModel = Args.getLastArgValue(OPT_mthread_model, "posix");
  if (Opts.ThreadModel != "posix" && Opts.ThreadModel != "single")
    Diags.Report(diag::err_drv_invalid_value)
        << Args.getLastArg(OPT_mthread_model)->getAsString(Args)
        << Opts.ThreadModel;
  Opts.TrapFuncName = Args.getLastArgValue(OPT_ftrap_function_EQ);
  Opts.UseInitArray = Args.hasArg(OPT_fuse_init_array);

  Opts.FunctionSections = Args.hasFlag(OPT_ffunction_sections,
                                       OPT_fno_function_sections, false);
  Opts.DataSections = Args.hasFlag(OPT_fdata_sections,
                                   OPT_fno_data_sections, false);
  Opts.UniqueSectionNames = Args.hasFlag(OPT_funique_section_names,
                                         OPT_fno_unique_section_names, true);

  Opts.MergeFunctions = Args.hasArg(OPT_fmerge_functions);

  Opts.PrepareForLTO = Args.hasArg(OPT_flto, OPT_flto_EQ);
  const Arg *A = Args.getLastArg(OPT_flto, OPT_flto_EQ);
  Opts.EmitSummaryIndex = A && A->containsValue("thin");
  if (Arg *A = Args.getLastArg(OPT_fthinlto_index_EQ)) {
    if (IK != IK_LLVM_IR)
      Diags.Report(diag::err_drv_argument_only_allowed_with)
          << A->getAsString(Args) << "-x ir";
    Opts.ThinLTOIndexFile = Args.getLastArgValue(OPT_fthinlto_index_EQ);
  }

  Opts.MSVolatile = Args.hasArg(OPT_fms_volatile);

  Opts.VectorizeBB = Args.hasArg(OPT_vectorize_slp_aggressive);
  Opts.VectorizeLoop = Args.hasArg(OPT_vectorize_loops);
  Opts.VectorizeSLP = Args.hasArg(OPT_vectorize_slp);

  Opts.MainFileName = Args.getLastArgValue(OPT_main_file_name);
  Opts.VerifyModule = !Args.hasArg(OPT_disable_llvm_verifier);

  Opts.DisableGCov = Args.hasArg(OPT_test_coverage);
  Opts.EmitGcovArcs = Args.hasArg(OPT_femit_coverage_data);
  Opts.EmitGcovNotes = Args.hasArg(OPT_femit_coverage_notes);
  if (Opts.EmitGcovArcs || Opts.EmitGcovNotes) {
    Opts.CoverageFile = Args.getLastArgValue(OPT_coverage_file);
    Opts.CoverageExtraChecksum = Args.hasArg(OPT_coverage_cfg_checksum);
    Opts.CoverageNoFunctionNamesInData =
        Args.hasArg(OPT_coverage_no_function_names_in_data);
    Opts.CoverageExitBlockBeforeBody =
        Args.hasArg(OPT_coverage_exit_block_before_body);
    if (Args.hasArg(OPT_coverage_version_EQ)) {
      StringRef CoverageVersion = Args.getLastArgValue(OPT_coverage_version_EQ);
      if (CoverageVersion.size() != 4) {
        Diags.Report(diag::err_drv_invalid_value)
            << Args.getLastArg(OPT_coverage_version_EQ)->getAsString(Args)
            << CoverageVersion;
      } else {
        memcpy(Opts.CoverageVersion, CoverageVersion.data(), 4);
      }
    }
  }

  Opts.InstrumentFunctions = Args.hasArg(OPT_finstrument_functions);
  Opts.InstrumentForProfiling = Args.hasArg(OPT_pg);
  Opts.EmitOpenCLArgMetadata = Args.hasArg(OPT_cl_kernel_arg_info);
  Opts.CompressDebugSections = Args.hasArg(OPT_compress_debug_sections);
  Opts.DebugCompilationDir = Args.getLastArgValue(OPT_fdebug_compilation_dir);
  for (auto A : Args.filtered(OPT_mlink_bitcode_file, OPT_mlink_cuda_bitcode)) {
    unsigned LinkFlags = llvm::Linker::Flags::None;
    if (A->getOption().matches(OPT_mlink_cuda_bitcode))
      LinkFlags = llvm::Linker::Flags::LinkOnlyNeeded |
                  llvm::Linker::Flags::InternalizeLinkedSymbols;
    Opts.LinkBitcodeFiles.push_back(std::make_pair(LinkFlags, A->getValue()));
  }
  Opts.SanitizeCoverageType =
      getLastArgIntValue(Args, OPT_fsanitize_coverage_type, 0, Diags);
  Opts.SanitizeCoverageIndirectCalls =
      Args.hasArg(OPT_fsanitize_coverage_indirect_calls);
  Opts.SanitizeCoverageTraceBB = Args.hasArg(OPT_fsanitize_coverage_trace_bb);
  Opts.SanitizeCoverageTraceCmp = Args.hasArg(OPT_fsanitize_coverage_trace_cmp);
  Opts.SanitizeCoverage8bitCounters =
      Args.hasArg(OPT_fsanitize_coverage_8bit_counters);
  Opts.SanitizeMemoryTrackOrigins =
      getLastArgIntValue(Args, OPT_fsanitize_memory_track_origins_EQ, 0, Diags);
  Opts.SanitizeMemoryUseAfterDtor =
      Args.hasArg(OPT_fsanitize_memory_use_after_dtor);
  Opts.SanitizeCfiCrossDso = Args.hasArg(OPT_fsanitize_cfi_cross_dso);
  Opts.SanitizeStats = Args.hasArg(OPT_fsanitize_stats);
  Opts.SSPBufferSize =
      getLastArgIntValue(Args, OPT_stack_protector_buffer_size, 8, Diags);
  Opts.StackRealignment = Args.hasArg(OPT_mstackrealign);
  if (Arg *A = Args.getLastArg(OPT_mstack_alignment)) {
    StringRef Val = A->getValue();
    unsigned StackAlignment = Opts.StackAlignment;
    Val.getAsInteger(10, StackAlignment);
    Opts.StackAlignment = StackAlignment;
  }

  if (Arg *A = Args.getLastArg(OPT_mstack_probe_size)) {
    StringRef Val = A->getValue();
    unsigned StackProbeSize = Opts.StackProbeSize;
    Val.getAsInteger(0, StackProbeSize);
    Opts.StackProbeSize = StackProbeSize;
  }

  if (Arg *A = Args.getLastArg(OPT_fobjc_dispatch_method_EQ)) {
    StringRef Name = A->getValue();
    unsigned Method = llvm::StringSwitch<unsigned>(Name)
      .Case("legacy", CodeGenOptions::Legacy)
      .Case("non-legacy", CodeGenOptions::NonLegacy)
      .Case("mixed", CodeGenOptions::Mixed)
      .Default(~0U);
    if (Method == ~0U) {
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.setObjCDispatchMethod(
        static_cast<CodeGenOptions::ObjCDispatchMethodKind>(Method));
    }
  }

  Opts.EmulatedTLS =
      Args.hasFlag(OPT_femulated_tls, OPT_fno_emulated_tls, false);

  if (Arg *A = Args.getLastArg(OPT_ftlsmodel_EQ)) {
    StringRef Name = A->getValue();
    unsigned Model = llvm::StringSwitch<unsigned>(Name)
        .Case("global-dynamic", CodeGenOptions::GeneralDynamicTLSModel)
        .Case("local-dynamic", CodeGenOptions::LocalDynamicTLSModel)
        .Case("initial-exec", CodeGenOptions::InitialExecTLSModel)
        .Case("local-exec", CodeGenOptions::LocalExecTLSModel)
        .Default(~0U);
    if (Model == ~0U) {
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Name;
      Success = false;
    } else {
      Opts.setDefaultTLSModel(static_cast<CodeGenOptions::TLSModel>(Model));
    }
  }

  if (Arg *A = Args.getLastArg(OPT_ffp_contract)) {
    StringRef Val = A->getValue();
    if (Val == "fast")
      Opts.setFPContractMode(CodeGenOptions::FPC_Fast);
    else if (Val == "on")
      Opts.setFPContractMode(CodeGenOptions::FPC_On);
    else if (Val == "off")
      Opts.setFPContractMode(CodeGenOptions::FPC_Off);
    else
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Val;
  }

  if (Arg *A = Args.getLastArg(OPT_fpcc_struct_return, OPT_freg_struct_return)) {
    if (A->getOption().matches(OPT_fpcc_struct_return)) {
      Opts.setStructReturnConvention(CodeGenOptions::SRCK_OnStack);
    } else {
      assert(A->getOption().matches(OPT_freg_struct_return));
      Opts.setStructReturnConvention(CodeGenOptions::SRCK_InRegs);
    }
  }

  Opts.DependentLibraries = Args.getAllArgValues(OPT_dependent_lib);
  Opts.LinkerOptions = Args.getAllArgValues(OPT_linker_option);
  bool NeedLocTracking = false;

  if (Arg *A = Args.getLastArg(OPT_Rpass_EQ)) {
    Opts.OptimizationRemarkPattern =
        GenerateOptimizationRemarkRegex(Diags, Args, A);
    NeedLocTracking = true;
  }

  if (Arg *A = Args.getLastArg(OPT_Rpass_missed_EQ)) {
    Opts.OptimizationRemarkMissedPattern =
        GenerateOptimizationRemarkRegex(Diags, Args, A);
    NeedLocTracking = true;
  }

  if (Arg *A = Args.getLastArg(OPT_Rpass_analysis_EQ)) {
    Opts.OptimizationRemarkAnalysisPattern =
        GenerateOptimizationRemarkRegex(Diags, Args, A);
    NeedLocTracking = true;
  }

  // If the user requested to use a sample profile for PGO, then the
  // backend will need to track source location information so the profile
  // can be incorporated into the IR.
  if (!Opts.SampleProfileFile.empty())
    NeedLocTracking = true;

  // If the user requested a flag that requires source locations available in
  // the backend, make sure that the backend tracks source location information.
  if (NeedLocTracking && Opts.getDebugInfo() == codegenoptions::NoDebugInfo)
    Opts.setDebugInfo(codegenoptions::LocTrackingOnly);

  Opts.RewriteMapFiles = Args.getAllArgValues(OPT_frewrite_map_file);

  // Parse -fsanitize-recover= arguments.
  // FIXME: Report unrecoverable sanitizers incorrectly specified here.
  parseSanitizerKinds("-fsanitize-recover=",
                      Args.getAllArgValues(OPT_fsanitize_recover_EQ), Diags,
                      Opts.SanitizeRecover);
  parseSanitizerKinds("-fsanitize-trap=",
                      Args.getAllArgValues(OPT_fsanitize_trap_EQ), Diags,
                      Opts.SanitizeTrap);

  Opts.CudaGpuBinaryFileNames =
      Args.getAllArgValues(OPT_fcuda_include_gpubinary);

  return Success;
}

static void ParseDependencyOutputArgs(DependencyOutputOptions &Opts,
                                      ArgList &Args) {
  using namespace options;
  Opts.OutputFile = Args.getLastArgValue(OPT_dependency_file);
  Opts.Targets = Args.getAllArgValues(OPT_MT);
  Opts.IncludeSystemHeaders = Args.hasArg(OPT_sys_header_deps);
  Opts.IncludeModuleFiles = Args.hasArg(OPT_module_file_deps);
  Opts.UsePhonyTargets = Args.hasArg(OPT_MP);
  Opts.ShowHeaderIncludes = Args.hasArg(OPT_H);
  Opts.HeaderIncludeOutputFile = Args.getLastArgValue(OPT_header_include_file);
  Opts.AddMissingHeaderDeps = Args.hasArg(OPT_MG);
  Opts.PrintShowIncludes = Args.hasArg(OPT_show_includes);
  Opts.DOTOutputFile = Args.getLastArgValue(OPT_dependency_dot);
  Opts.ModuleDependencyOutputDir =
      Args.getLastArgValue(OPT_module_dependency_dir);
  if (Args.hasArg(OPT_MV))
    Opts.OutputFormat = DependencyOutputFormat::NMake;
  // Add sanitizer blacklists as extra dependencies.
  // They won't be discovered by the regular preprocessor, so
  // we let make / ninja to know about this implicit dependency.
  Opts.ExtraDeps = Args.getAllArgValues(OPT_fdepfile_entry);
  auto ModuleFiles = Args.getAllArgValues(OPT_fmodule_file);
  Opts.ExtraDeps.insert(Opts.ExtraDeps.end(), ModuleFiles.begin(),
                        ModuleFiles.end());
}

static bool parseShowColorsArgs(const ArgList &Args, bool DefaultColor) {
  // Color diagnostics default to auto ("on" if terminal supports) in the driver
  // but default to off in cc1, needing an explicit OPT_fdiagnostics_color.
  // Support both clang's -f[no-]color-diagnostics and gcc's
  // -f[no-]diagnostics-colors[=never|always|auto].
  enum {
    Colors_On,
    Colors_Off,
    Colors_Auto
  } ShowColors = DefaultColor ? Colors_Auto : Colors_Off;
  for (Arg *A : Args) {
    const Option &O = A->getOption();
    if (!O.matches(options::OPT_fcolor_diagnostics) &&
        !O.matches(options::OPT_fdiagnostics_color) &&
        !O.matches(options::OPT_fno_color_diagnostics) &&
        !O.matches(options::OPT_fno_diagnostics_color) &&
        !O.matches(options::OPT_fdiagnostics_color_EQ))
      continue;

    if (O.matches(options::OPT_fcolor_diagnostics) ||
        O.matches(options::OPT_fdiagnostics_color)) {
      ShowColors = Colors_On;
    } else if (O.matches(options::OPT_fno_color_diagnostics) ||
               O.matches(options::OPT_fno_diagnostics_color)) {
      ShowColors = Colors_Off;
    } else {
      assert(O.matches(options::OPT_fdiagnostics_color_EQ));
      StringRef Value(A->getValue());
      if (Value == "always")
        ShowColors = Colors_On;
      else if (Value == "never")
        ShowColors = Colors_Off;
      else if (Value == "auto")
        ShowColors = Colors_Auto;
    }
  }
  if (ShowColors == Colors_On ||
      (ShowColors == Colors_Auto && llvm::sys::Process::StandardErrHasColors()))
    return true;
  return false;
}

bool clang::ParseDiagnosticArgs(DiagnosticOptions &Opts, ArgList &Args,
                                DiagnosticsEngine *Diags,
                                bool DefaultDiagColor) {
  using namespace options;
  bool Success = true;

  Opts.DiagnosticLogFile = Args.getLastArgValue(OPT_diagnostic_log_file);
  if (Arg *A =
          Args.getLastArg(OPT_diagnostic_serialized_file, OPT__serialize_diags))
    Opts.DiagnosticSerializationFile = A->getValue();
  Opts.IgnoreWarnings = Args.hasArg(OPT_w);
  Opts.NoRewriteMacros = Args.hasArg(OPT_Wno_rewrite_macros);
  Opts.Pedantic = Args.hasArg(OPT_pedantic);
  Opts.PedanticErrors = Args.hasArg(OPT_pedantic_errors);
  Opts.ShowCarets = !Args.hasArg(OPT_fno_caret_diagnostics);
  Opts.ShowColors = parseShowColorsArgs(Args, DefaultDiagColor);
  Opts.ShowColumn = Args.hasFlag(OPT_fshow_column,
                                 OPT_fno_show_column,
                                 /*Default=*/true);
  Opts.ShowFixits = !Args.hasArg(OPT_fno_diagnostics_fixit_info);
  Opts.ShowLocation = !Args.hasArg(OPT_fno_show_source_location);
  Opts.ShowOptionNames = Args.hasArg(OPT_fdiagnostics_show_option);

  llvm::sys::Process::UseANSIEscapeCodes(Args.hasArg(OPT_fansi_escape_codes));

  // Default behavior is to not to show note include stacks.
  Opts.ShowNoteIncludeStack = false;
  if (Arg *A = Args.getLastArg(OPT_fdiagnostics_show_note_include_stack,
                               OPT_fno_diagnostics_show_note_include_stack))
    if (A->getOption().matches(OPT_fdiagnostics_show_note_include_stack))
      Opts.ShowNoteIncludeStack = true;

  StringRef ShowOverloads =
    Args.getLastArgValue(OPT_fshow_overloads_EQ, "all");
  if (ShowOverloads == "best")
    Opts.setShowOverloads(Ovl_Best);
  else if (ShowOverloads == "all")
    Opts.setShowOverloads(Ovl_All);
  else {
    Success = false;
    if (Diags)
      Diags->Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_fshow_overloads_EQ)->getAsString(Args)
      << ShowOverloads;
  }

  StringRef ShowCategory =
    Args.getLastArgValue(OPT_fdiagnostics_show_category, "none");
  if (ShowCategory == "none")
    Opts.ShowCategories = 0;
  else if (ShowCategory == "id")
    Opts.ShowCategories = 1;
  else if (ShowCategory == "name")
    Opts.ShowCategories = 2;
  else {
    Success = false;
    if (Diags)
      Diags->Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_fdiagnostics_show_category)->getAsString(Args)
      << ShowCategory;
  }

  StringRef Format =
    Args.getLastArgValue(OPT_fdiagnostics_format, "clang");
  if (Format == "clang")
    Opts.setFormat(DiagnosticOptions::Clang);
  else if (Format == "msvc")
    Opts.setFormat(DiagnosticOptions::MSVC);
  else if (Format == "msvc-fallback") {
    Opts.setFormat(DiagnosticOptions::MSVC);
    Opts.CLFallbackMode = true;
  } else if (Format == "vi")
    Opts.setFormat(DiagnosticOptions::Vi);
  else {
    Success = false;
    if (Diags)
      Diags->Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_fdiagnostics_format)->getAsString(Args)
      << Format;
  }

  Opts.ShowSourceRanges = Args.hasArg(OPT_fdiagnostics_print_source_range_info);
  Opts.ShowParseableFixits = Args.hasArg(OPT_fdiagnostics_parseable_fixits);
  Opts.ShowPresumedLoc = !Args.hasArg(OPT_fno_diagnostics_use_presumed_location);
  Opts.VerifyDiagnostics = Args.hasArg(OPT_verify);
  DiagnosticLevelMask DiagMask = DiagnosticLevelMask::None;
  Success &= parseDiagnosticLevelMask("-verify-ignore-unexpected=",
    Args.getAllArgValues(OPT_verify_ignore_unexpected_EQ),
    Diags, DiagMask);
  if (Args.hasArg(OPT_verify_ignore_unexpected))
    DiagMask = DiagnosticLevelMask::All;
  Opts.setVerifyIgnoreUnexpected(DiagMask);
  Opts.ElideType = !Args.hasArg(OPT_fno_elide_type);
  Opts.ShowTemplateTree = Args.hasArg(OPT_fdiagnostics_show_template_tree);
  Opts.ErrorLimit = getLastArgIntValue(Args, OPT_ferror_limit, 0, Diags);
  Opts.MacroBacktraceLimit =
      getLastArgIntValue(Args, OPT_fmacro_backtrace_limit,
                         DiagnosticOptions::DefaultMacroBacktraceLimit, Diags);
  Opts.TemplateBacktraceLimit = getLastArgIntValue(
      Args, OPT_ftemplate_backtrace_limit,
      DiagnosticOptions::DefaultTemplateBacktraceLimit, Diags);
  Opts.ConstexprBacktraceLimit = getLastArgIntValue(
      Args, OPT_fconstexpr_backtrace_limit,
      DiagnosticOptions::DefaultConstexprBacktraceLimit, Diags);
  Opts.SpellCheckingLimit = getLastArgIntValue(
      Args, OPT_fspell_checking_limit,
      DiagnosticOptions::DefaultSpellCheckingLimit, Diags);
  Opts.TabStop = getLastArgIntValue(Args, OPT_ftabstop,
                                    DiagnosticOptions::DefaultTabStop, Diags);
  if (Opts.TabStop == 0 || Opts.TabStop > DiagnosticOptions::MaxTabStop) {
    Opts.TabStop = DiagnosticOptions::DefaultTabStop;
    if (Diags)
      Diags->Report(diag::warn_ignoring_ftabstop_value)
      << Opts.TabStop << DiagnosticOptions::DefaultTabStop;
  }
  Opts.MessageLength = getLastArgIntValue(Args, OPT_fmessage_length, 0, Diags);
  addDiagnosticArgs(Args, OPT_W_Group, OPT_W_value_Group, Opts.Warnings);
  addDiagnosticArgs(Args, OPT_R_Group, OPT_R_value_Group, Opts.Remarks);

  return Success;
}

static void ParseFileSystemArgs(FileSystemOptions &Opts, ArgList &Args) {
  Opts.WorkingDir = Args.getLastArgValue(OPT_working_directory);
  Opts.APINotesCachePath = Args.getLastArgValue(OPT_fapinotes_cache_path);
}

/// Parse the argument to the -ftest-module-file-extension
/// command-line argument.
///
/// \returns true on error, false on success.
static bool parseTestModuleFileExtensionArg(StringRef Arg,
                                            std::string &BlockName,
                                            unsigned &MajorVersion,
                                            unsigned &MinorVersion,
                                            bool &Hashed,
                                            std::string &UserInfo) {
  SmallVector<StringRef, 5> Args;
  Arg.split(Args, ':', 5);
  if (Args.size() < 5)
    return true;

  BlockName = Args[0];
  if (Args[1].getAsInteger(10, MajorVersion)) return true;
  if (Args[2].getAsInteger(10, MinorVersion)) return true;
  if (Args[3].getAsInteger(2, Hashed)) return true;
  if (Args.size() > 4)
    UserInfo = Args[4];
  return false;
}

static InputKind ParseFrontendArgs(FrontendOptions &Opts, ArgList &Args,
                                   DiagnosticsEngine &Diags) {
  using namespace options;
  Opts.ProgramAction = frontend::ParseSyntaxOnly;
  if (const Arg *A = Args.getLastArg(OPT_Action_Group)) {
    switch (A->getOption().getID()) {
    default:
      llvm_unreachable("Invalid option in group!");
    case OPT_ast_list:
      Opts.ProgramAction = frontend::ASTDeclList; break;
    case OPT_ast_dump:
    case OPT_ast_dump_lookups:
      Opts.ProgramAction = frontend::ASTDump; break;
    case OPT_ast_print:
      Opts.ProgramAction = frontend::ASTPrint; break;
    case OPT_ast_view:
      Opts.ProgramAction = frontend::ASTView; break;
    case OPT_dump_raw_tokens:
      Opts.ProgramAction = frontend::DumpRawTokens; break;
    case OPT_dump_tokens:
      Opts.ProgramAction = frontend::DumpTokens; break;
    case OPT_S:
      Opts.ProgramAction = frontend::EmitAssembly; break;
    case OPT_emit_llvm_bc:
      Opts.ProgramAction = frontend::EmitBC; break;
    case OPT_emit_html:
      Opts.ProgramAction = frontend::EmitHTML; break;
    case OPT_emit_llvm:
      Opts.ProgramAction = frontend::EmitLLVM; break;
    case OPT_emit_llvm_only:
      Opts.ProgramAction = frontend::EmitLLVMOnly; break;
    case OPT_emit_codegen_only:
      Opts.ProgramAction = frontend::EmitCodeGenOnly; break;
    case OPT_emit_obj:
      Opts.ProgramAction = frontend::EmitObj; break;
    case OPT_fixit_EQ:
      Opts.FixItSuffix = A->getValue();
      // fall-through!
    case OPT_fixit:
      Opts.ProgramAction = frontend::FixIt; break;
    case OPT_emit_module:
      Opts.ProgramAction = frontend::GenerateModule; break;
    case OPT_emit_pch:
      Opts.ProgramAction = frontend::GeneratePCH; break;
    case OPT_emit_pth:
      Opts.ProgramAction = frontend::GeneratePTH; break;
    case OPT_init_only:
      Opts.ProgramAction = frontend::InitOnly; break;
    case OPT_fsyntax_only:
      Opts.ProgramAction = frontend::ParseSyntaxOnly; break;
    case OPT_module_file_info:
      Opts.ProgramAction = frontend::ModuleFileInfo; break;
    case OPT_verify_pch:
      Opts.ProgramAction = frontend::VerifyPCH; break;
    case OPT_print_decl_contexts:
      Opts.ProgramAction = frontend::PrintDeclContext; break;
    case OPT_print_preamble:
      Opts.ProgramAction = frontend::PrintPreamble; break;
    case OPT_E:
      Opts.ProgramAction = frontend::PrintPreprocessedInput; break;
    case OPT_rewrite_macros:
      Opts.ProgramAction = frontend::RewriteMacros; break;
    case OPT_rewrite_objc:
      Opts.ProgramAction = frontend::RewriteObjC; break;
    case OPT_rewrite_test:
      Opts.ProgramAction = frontend::RewriteTest; break;
    case OPT_analyze:
      Opts.ProgramAction = frontend::RunAnalysis; break;
    case OPT_migrate:
      Opts.ProgramAction = frontend::MigrateSource; break;
    case OPT_Eonly:
      Opts.ProgramAction = frontend::RunPreprocessorOnly; break;
    }
  }

  if (const Arg* A = Args.getLastArg(OPT_plugin)) {
    Opts.Plugins.emplace_back(A->getValue(0));
    Opts.ProgramAction = frontend::PluginAction;
    Opts.ActionName = A->getValue();

    for (const Arg *AA : Args.filtered(OPT_plugin_arg))
      if (AA->getValue(0) == Opts.ActionName)
        Opts.PluginArgs.emplace_back(AA->getValue(1));
  }

  Opts.AddPluginActions = Args.getAllArgValues(OPT_add_plugin);
  Opts.AddPluginArgs.resize(Opts.AddPluginActions.size());
  for (int i = 0, e = Opts.AddPluginActions.size(); i != e; ++i)
    for (const Arg *A : Args.filtered(OPT_plugin_arg))
      if (A->getValue(0) == Opts.AddPluginActions[i])
        Opts.AddPluginArgs[i].emplace_back(A->getValue(1));

  for (const std::string &Arg :
         Args.getAllArgValues(OPT_ftest_module_file_extension_EQ)) {
    std::string BlockName;
    unsigned MajorVersion;
    unsigned MinorVersion;
    bool Hashed;
    std::string UserInfo;
    if (parseTestModuleFileExtensionArg(Arg, BlockName, MajorVersion,
                                        MinorVersion, Hashed, UserInfo)) {
      Diags.Report(diag::err_test_module_file_extension_format) << Arg;

      continue;
    }

    // Add the testing module file extension.
    Opts.ModuleFileExtensions.push_back(
      new TestModuleFileExtension(BlockName, MajorVersion, MinorVersion,
                                  Hashed, UserInfo));
  }

  if (const Arg *A = Args.getLastArg(OPT_code_completion_at)) {
    Opts.CodeCompletionAt =
      ParsedSourceLocation::FromString(A->getValue());
    if (Opts.CodeCompletionAt.FileName.empty())
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue();
  }
  Opts.DisableFree = Args.hasArg(OPT_disable_free);

  Opts.OutputFile = Args.getLastArgValue(OPT_o);
  Opts.Plugins = Args.getAllArgValues(OPT_load);
  Opts.RelocatablePCH = Args.hasArg(OPT_relocatable_pch);
  Opts.ShowHelp = Args.hasArg(OPT_help);
  Opts.ShowStats = Args.hasArg(OPT_print_stats);
  Opts.ShowTimers = Args.hasArg(OPT_ftime_report);
  Opts.ShowVersion = Args.hasArg(OPT_version);
  Opts.ASTMergeFiles = Args.getAllArgValues(OPT_ast_merge);
  Opts.LLVMArgs = Args.getAllArgValues(OPT_mllvm);
  Opts.FixWhatYouCan = Args.hasArg(OPT_fix_what_you_can);
  Opts.FixOnlyWarnings = Args.hasArg(OPT_fix_only_warnings);
  Opts.FixAndRecompile = Args.hasArg(OPT_fixit_recompile);
  Opts.FixToTemporaries = Args.hasArg(OPT_fixit_to_temp);
  Opts.ASTDumpDecls = Args.hasArg(OPT_ast_dump);
  Opts.ASTDumpFilter = Args.getLastArgValue(OPT_ast_dump_filter);
  Opts.ASTDumpLookups = Args.hasArg(OPT_ast_dump_lookups);
  Opts.UseGlobalModuleIndex = !Args.hasArg(OPT_fno_modules_global_index);
  Opts.GenerateGlobalModuleIndex = Opts.UseGlobalModuleIndex;
  Opts.ModuleMapFiles = Args.getAllArgValues(OPT_fmodule_map_file);
  Opts.ModuleFiles = Args.getAllArgValues(OPT_fmodule_file);
  Opts.ModulesEmbedFiles = Args.getAllArgValues(OPT_fmodules_embed_file_EQ);
  Opts.ModulesEmbedAllFiles = Args.hasArg(OPT_fmodules_embed_all_files);

  Opts.CodeCompleteOpts.IncludeMacros
    = Args.hasArg(OPT_code_completion_macros);
  Opts.CodeCompleteOpts.IncludeCodePatterns
    = Args.hasArg(OPT_code_completion_patterns);
  Opts.CodeCompleteOpts.IncludeGlobals
    = !Args.hasArg(OPT_no_code_completion_globals);
  Opts.CodeCompleteOpts.IncludeBriefComments
    = Args.hasArg(OPT_code_completion_brief_comments);

  Opts.OverrideRecordLayoutsFile
    = Args.getLastArgValue(OPT_foverride_record_layout_EQ);
  Opts.AuxTriple =
      llvm::Triple::normalize(Args.getLastArgValue(OPT_aux_triple));

  if (const Arg *A = Args.getLastArg(OPT_arcmt_check,
                                     OPT_arcmt_modify,
                                     OPT_arcmt_migrate)) {
    switch (A->getOption().getID()) {
    default:
      llvm_unreachable("missed a case");
    case OPT_arcmt_check:
      Opts.ARCMTAction = FrontendOptions::ARCMT_Check;
      break;
    case OPT_arcmt_modify:
      Opts.ARCMTAction = FrontendOptions::ARCMT_Modify;
      break;
    case OPT_arcmt_migrate:
      Opts.ARCMTAction = FrontendOptions::ARCMT_Migrate;
      break;
    }
  }
  Opts.MTMigrateDir = Args.getLastArgValue(OPT_mt_migrate_directory);
  Opts.ARCMTMigrateReportOut
    = Args.getLastArgValue(OPT_arcmt_migrate_report_output);
  Opts.ARCMTMigrateEmitARCErrors
    = Args.hasArg(OPT_arcmt_migrate_emit_arc_errors);

  if (Args.hasArg(OPT_objcmt_migrate_literals))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_Literals;
  if (Args.hasArg(OPT_objcmt_migrate_subscripting))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_Subscripting;
  if (Args.hasArg(OPT_objcmt_migrate_property_dot_syntax))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_PropertyDotSyntax;
  if (Args.hasArg(OPT_objcmt_migrate_property))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_Property;
  if (Args.hasArg(OPT_objcmt_migrate_readonly_property))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_ReadonlyProperty;
  if (Args.hasArg(OPT_objcmt_migrate_readwrite_property))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_ReadwriteProperty;
  if (Args.hasArg(OPT_objcmt_migrate_annotation))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_Annotation;
  if (Args.hasArg(OPT_objcmt_returns_innerpointer_property))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_ReturnsInnerPointerProperty;
  if (Args.hasArg(OPT_objcmt_migrate_instancetype))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_Instancetype;
  if (Args.hasArg(OPT_objcmt_migrate_nsmacros))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_NsMacros;
  if (Args.hasArg(OPT_objcmt_migrate_protocol_conformance))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_ProtocolConformance;
  if (Args.hasArg(OPT_objcmt_atomic_property))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_AtomicProperty;
  if (Args.hasArg(OPT_objcmt_ns_nonatomic_iosonly))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_NsAtomicIOSOnlyProperty;
  if (Args.hasArg(OPT_objcmt_migrate_designated_init))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_DesignatedInitializer;
  if (Args.hasArg(OPT_objcmt_migrate_all))
    Opts.ObjCMTAction |= FrontendOptions::ObjCMT_MigrateDecls;

  Opts.ObjCMTWhiteListPath = Args.getLastArgValue(OPT_objcmt_whitelist_dir_path);

  if (Opts.ARCMTAction != FrontendOptions::ARCMT_None &&
      Opts.ObjCMTAction != FrontendOptions::ObjCMT_None) {
    Diags.Report(diag::err_drv_argument_not_allowed_with)
      << "ARC migration" << "ObjC migration";
  }

  InputKind DashX = IK_None;
  if (const Arg *A = Args.getLastArg(OPT_x)) {
    DashX = llvm::StringSwitch<InputKind>(A->getValue())
      .Case("c", IK_C)
      .Case("cl", IK_OpenCL)
      .Case("cuda", IK_CUDA)
      .Case("c++", IK_CXX)
      .Case("objective-c", IK_ObjC)
      .Case("objective-c++", IK_ObjCXX)
      .Case("cpp-output", IK_PreprocessedC)
      .Case("assembler-with-cpp", IK_Asm)
      .Case("c++-cpp-output", IK_PreprocessedCXX)
      .Case("cuda-cpp-output", IK_PreprocessedCuda)
      .Case("objective-c-cpp-output", IK_PreprocessedObjC)
      .Case("objc-cpp-output", IK_PreprocessedObjC)
      .Case("objective-c++-cpp-output", IK_PreprocessedObjCXX)
      .Case("objc++-cpp-output", IK_PreprocessedObjCXX)
      .Case("c-header", IK_C)
      .Case("cl-header", IK_OpenCL)
      .Case("objective-c-header", IK_ObjC)
      .Case("c++-header", IK_CXX)
      .Case("objective-c++-header", IK_ObjCXX)
      .Cases("ast", "pcm", IK_AST)
      .Case("ir", IK_LLVM_IR)
      .Default(IK_None);
    if (DashX == IK_None)
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue();
  }

  // '-' is the default input if none is given.
  std::vector<std::string> Inputs = Args.getAllArgValues(OPT_INPUT);
  Opts.Inputs.clear();
  if (Inputs.empty())
    Inputs.push_back("-");
  for (unsigned i = 0, e = Inputs.size(); i != e; ++i) {
    InputKind IK = DashX;
    if (IK == IK_None) {
      IK = FrontendOptions::getInputKindForExtension(
        StringRef(Inputs[i]).rsplit('.').second);
      // FIXME: Remove this hack.
      if (i == 0)
        DashX = IK;
    }
    Opts.Inputs.emplace_back(std::move(Inputs[i]), IK);
  }

  return DashX;
}

std::string CompilerInvocation::GetResourcesPath(const char *Argv0,
                                                 void *MainAddr) {
  std::string ClangExecutable =
      llvm::sys::fs::getMainExecutable(Argv0, MainAddr);
  StringRef Dir = llvm::sys::path::parent_path(ClangExecutable);

  // Compute the path to the resource directory.
  StringRef ClangResourceDir(CLANG_RESOURCE_DIR);
  SmallString<128> P(Dir);
  if (ClangResourceDir != "")
    llvm::sys::path::append(P, ClangResourceDir);
  else
    llvm::sys::path::append(P, "..", Twine("lib") + CLANG_LIBDIR_SUFFIX,
                            "clang", CLANG_VERSION_STRING);

  return P.str();
}

static void ParseHeaderSearchArgs(HeaderSearchOptions &Opts, ArgList &Args) {
  using namespace options;
  Opts.Sysroot = Args.getLastArgValue(OPT_isysroot, "/");
  Opts.Verbose = Args.hasArg(OPT_v);
  Opts.UseBuiltinIncludes = !Args.hasArg(OPT_nobuiltininc);
  Opts.UseStandardSystemIncludes = !Args.hasArg(OPT_nostdsysteminc);
  Opts.UseStandardCXXIncludes = !Args.hasArg(OPT_nostdincxx);
  if (const Arg *A = Args.getLastArg(OPT_stdlib_EQ))
    Opts.UseLibcxx = (strcmp(A->getValue(), "libc++") == 0);
  Opts.ResourceDir = Args.getLastArgValue(OPT_resource_dir);
  Opts.ModuleCachePath = Args.getLastArgValue(OPT_fmodules_cache_path);
  Opts.ModuleUserBuildPath = Args.getLastArgValue(OPT_fmodules_user_build_path);
  for (const Arg *A : Args.filtered(OPT_fprebuilt_module_path))
    Opts.AddPrebuiltModulePath(A->getValue());
  Opts.DisableModuleHash = Args.hasArg(OPT_fdisable_module_hash);
  Opts.ModulesValidateDiagnosticOptions =
      !Args.hasArg(OPT_fmodules_disable_diagnostic_validation);
  Opts.ImplicitModuleMaps = Args.hasArg(OPT_fimplicit_module_maps);
  Opts.ModuleMapFileHomeIsCwd = Args.hasArg(OPT_fmodule_map_file_home_is_cwd);
  Opts.ModuleCachePruneInterval =
      getLastArgIntValue(Args, OPT_fmodules_prune_interval, 7 * 24 * 60 * 60);
  Opts.ModuleCachePruneAfter =
      getLastArgIntValue(Args, OPT_fmodules_prune_after, 31 * 24 * 60 * 60);
  Opts.ModulesValidateOncePerBuildSession =
      Args.hasArg(OPT_fmodules_validate_once_per_build_session);
  Opts.BuildSessionTimestamp =
      getLastArgUInt64Value(Args, OPT_fbuild_session_timestamp, 0);
  Opts.ModulesValidateSystemHeaders =
      Args.hasArg(OPT_fmodules_validate_system_headers);
  if (const Arg *A = Args.getLastArg(OPT_fmodule_format_EQ))
    Opts.ModuleFormat = A->getValue();

  for (const Arg *A : Args.filtered(OPT_fmodules_ignore_macro)) {
    StringRef MacroDef = A->getValue();
    Opts.ModulesIgnoreMacros.insert(MacroDef.split('=').first);
  }

  // Add -I..., -F..., and -index-header-map options in order.
  bool IsIndexHeaderMap = false;
  for (const Arg *A : Args.filtered(OPT_I, OPT_F, OPT_index_header_map)) {
    if (A->getOption().matches(OPT_index_header_map)) {
      // -index-header-map applies to the next -I or -F.
      IsIndexHeaderMap = true;
      continue;
    }

    frontend::IncludeDirGroup Group =
        IsIndexHeaderMap ? frontend::IndexHeaderMap : frontend::Angled;

    Opts.AddPath(A->getValue(), Group,
                 /*IsFramework=*/A->getOption().matches(OPT_F), true);
    IsIndexHeaderMap = false;
  }

  // Add -iprefix/-iwithprefix/-iwithprefixbefore options.
  StringRef Prefix = ""; // FIXME: This isn't the correct default prefix.
  for (const Arg *A :
       Args.filtered(OPT_iprefix, OPT_iwithprefix, OPT_iwithprefixbefore)) {
    if (A->getOption().matches(OPT_iprefix))
      Prefix = A->getValue();
    else if (A->getOption().matches(OPT_iwithprefix))
      Opts.AddPath(Prefix.str() + A->getValue(), frontend::After, false, true);
    else
      Opts.AddPath(Prefix.str() + A->getValue(), frontend::Angled, false, true);
  }

  for (const Arg *A : Args.filtered(OPT_idirafter))
    Opts.AddPath(A->getValue(), frontend::After, false, true);
  for (const Arg *A : Args.filtered(OPT_iquote))
    Opts.AddPath(A->getValue(), frontend::Quoted, false, true);
  for (const Arg *A : Args.filtered(OPT_isystem, OPT_iwithsysroot))
    Opts.AddPath(A->getValue(), frontend::System, false,
                 !A->getOption().matches(OPT_iwithsysroot));
  for (const Arg *A : Args.filtered(OPT_iframework))
    Opts.AddPath(A->getValue(), frontend::System, true, true);

  // Add the paths for the various language specific isystem flags.
  for (const Arg *A : Args.filtered(OPT_c_isystem))
    Opts.AddPath(A->getValue(), frontend::CSystem, false, true);
  for (const Arg *A : Args.filtered(OPT_cxx_isystem))
    Opts.AddPath(A->getValue(), frontend::CXXSystem, false, true);
  for (const Arg *A : Args.filtered(OPT_objc_isystem))
    Opts.AddPath(A->getValue(), frontend::ObjCSystem, false,true);
  for (const Arg *A : Args.filtered(OPT_objcxx_isystem))
    Opts.AddPath(A->getValue(), frontend::ObjCXXSystem, false, true);

  // Add the internal paths from a driver that detects standard include paths.
  for (const Arg *A :
       Args.filtered(OPT_internal_isystem, OPT_internal_externc_isystem)) {
    frontend::IncludeDirGroup Group = frontend::System;
    if (A->getOption().matches(OPT_internal_externc_isystem))
      Group = frontend::ExternCSystem;
    Opts.AddPath(A->getValue(), Group, false, true);
  }

  // Add the path prefixes which are implicitly treated as being system headers.
  for (const Arg *A :
       Args.filtered(OPT_system_header_prefix, OPT_no_system_header_prefix))
    Opts.AddSystemHeaderPrefix(
        A->getValue(), A->getOption().matches(OPT_system_header_prefix));

  for (const Arg *A : Args.filtered(OPT_ivfsoverlay))
    Opts.AddVFSOverlayFile(A->getValue());
}

static void ParseAPINotesArgs(APINotesOptions &Opts, ArgList &Args) {
  using namespace options;
  for (const Arg *A : Args.filtered(OPT_iapinotes_modules))
    Opts.ModuleSearchPaths.push_back(A->getValue());
}

void CompilerInvocation::setLangDefaults(LangOptions &Opts, InputKind IK,
                                         LangStandard::Kind LangStd) {
  // Set some properties which depend solely on the input kind; it would be nice
  // to move these to the language standard, and have the driver resolve the
  // input kind + language standard.
  if (IK == IK_Asm) {
    Opts.AsmPreprocessor = 1;
  } else if (IK == IK_ObjC ||
             IK == IK_ObjCXX ||
             IK == IK_PreprocessedObjC ||
             IK == IK_PreprocessedObjCXX) {
    Opts.ObjC1 = Opts.ObjC2 = 1;
  }

  if (LangStd == LangStandard::lang_unspecified) {
    // Based on the base language, pick one.
    switch (IK) {
    case IK_None:
    case IK_AST:
    case IK_LLVM_IR:
      llvm_unreachable("Invalid input kind!");
    case IK_OpenCL:
      LangStd = LangStandard::lang_opencl;
      break;
    case IK_CUDA:
    case IK_PreprocessedCuda:
      LangStd = LangStandard::lang_cuda;
      break;
    case IK_Asm:
    case IK_C:
    case IK_PreprocessedC:
    case IK_ObjC:
    case IK_PreprocessedObjC:
      LangStd = LangStandard::lang_gnu11;
      break;
    case IK_CXX:
    case IK_PreprocessedCXX:
    case IK_ObjCXX:
    case IK_PreprocessedObjCXX:
      LangStd = LangStandard::lang_gnucxx98;
      break;
    }
  }

  const LangStandard &Std = LangStandard::getLangStandardForKind(LangStd);
  Opts.LineComment = Std.hasLineComments();
  Opts.C99 = Std.isC99();
  Opts.C11 = Std.isC11();
  Opts.CPlusPlus = Std.isCPlusPlus();
  Opts.CPlusPlus11 = Std.isCPlusPlus11();
  Opts.CPlusPlus14 = Std.isCPlusPlus14();
  Opts.CPlusPlus1z = Std.isCPlusPlus1z();
  Opts.Digraphs = Std.hasDigraphs();
  Opts.GNUMode = Std.isGNUMode();
  Opts.GNUInline = Std.isC89();
  Opts.HexFloats = Std.hasHexFloats();
  Opts.ImplicitInt = Std.hasImplicitInt();

  // Set OpenCL Version.
  Opts.OpenCL = LangStd == LangStandard::lang_opencl || IK == IK_OpenCL;
  if (LangStd == LangStandard::lang_opencl)
    Opts.OpenCLVersion = 100;
  else if (LangStd == LangStandard::lang_opencl11)
    Opts.OpenCLVersion = 110;
  else if (LangStd == LangStandard::lang_opencl12)
    Opts.OpenCLVersion = 120;
  else if (LangStd == LangStandard::lang_opencl20)
    Opts.OpenCLVersion = 200;

  // OpenCL has some additional defaults.
  if (Opts.OpenCL) {
    Opts.AltiVec = 0;
    Opts.ZVector = 0;
    Opts.CXXOperatorNames = 1;
    Opts.LaxVectorConversions = 0;
    Opts.DefaultFPContract = 1;
    Opts.NativeHalfType = 1;
  }

  Opts.CUDA = IK == IK_CUDA || IK == IK_PreprocessedCuda ||
              LangStd == LangStandard::lang_cuda;

  // OpenCL and C++ both have bool, true, false keywords.
  Opts.Bool = Opts.OpenCL || Opts.CPlusPlus;

  // OpenCL has half keyword
  Opts.Half = Opts.OpenCL;

  // C++ has wchar_t keyword.
  Opts.WChar = Opts.CPlusPlus;

  Opts.GNUKeywords = Opts.GNUMode;
  Opts.CXXOperatorNames = Opts.CPlusPlus;

  Opts.DollarIdents = !Opts.AsmPreprocessor;
}

/// Attempt to parse a visibility value out of the given argument.
static Visibility parseVisibility(Arg *arg, ArgList &args,
                                  DiagnosticsEngine &diags) {
  StringRef value = arg->getValue();
  if (value == "default") {
    return DefaultVisibility;
  } else if (value == "hidden" || value == "internal") {
    return HiddenVisibility;
  } else if (value == "protected") {
    // FIXME: diagnose if target does not support protected visibility
    return ProtectedVisibility;
  }

  diags.Report(diag::err_drv_invalid_value)
    << arg->getAsString(args) << value;
  return DefaultVisibility;
}

static void ParseLangArgs(LangOptions &Opts, ArgList &Args, InputKind IK,
                          DiagnosticsEngine &Diags) {
  // FIXME: Cleanup per-file based stuff.
  LangStandard::Kind LangStd = LangStandard::lang_unspecified;
  if (const Arg *A = Args.getLastArg(OPT_std_EQ)) {
    LangStd = llvm::StringSwitch<LangStandard::Kind>(A->getValue())
#define LANGSTANDARD(id, name, desc, features) \
      .Case(name, LangStandard::lang_##id)
#include "clang/Frontend/LangStandards.def"
      .Default(LangStandard::lang_unspecified);
    if (LangStd == LangStandard::lang_unspecified)
      Diags.Report(diag::err_drv_invalid_value)
        << A->getAsString(Args) << A->getValue();
    else {
      // Valid standard, check to make sure language and standard are
      // compatible.
      const LangStandard &Std = LangStandard::getLangStandardForKind(LangStd);
      switch (IK) {
      case IK_C:
      case IK_ObjC:
      case IK_PreprocessedC:
      case IK_PreprocessedObjC:
        if (!(Std.isC89() || Std.isC99()))
          Diags.Report(diag::err_drv_argument_not_allowed_with)
            << A->getAsString(Args) << "C/ObjC";
        break;
      case IK_CXX:
      case IK_ObjCXX:
      case IK_PreprocessedCXX:
      case IK_PreprocessedObjCXX:
        if (!Std.isCPlusPlus())
          Diags.Report(diag::err_drv_argument_not_allowed_with)
            << A->getAsString(Args) << "C++/ObjC++";
        break;
      case IK_OpenCL:
        if (!Std.isC99())
          Diags.Report(diag::err_drv_argument_not_allowed_with)
            << A->getAsString(Args) << "OpenCL";
        break;
      case IK_CUDA:
      case IK_PreprocessedCuda:
        if (!Std.isCPlusPlus())
          Diags.Report(diag::err_drv_argument_not_allowed_with)
            << A->getAsString(Args) << "CUDA";
        break;
      default:
        break;
      }
    }
  }

  // -cl-std only applies for OpenCL language standards.
  // Override the -std option in this case.
  if (const Arg *A = Args.getLastArg(OPT_cl_std_EQ)) {
    LangStandard::Kind OpenCLLangStd
    = llvm::StringSwitch<LangStandard::Kind>(A->getValue())
    .Case("CL", LangStandard::lang_opencl)
    .Case("CL1.1", LangStandard::lang_opencl11)
    .Case("CL1.2", LangStandard::lang_opencl12)
    .Case("CL2.0", LangStandard::lang_opencl20)
    .Default(LangStandard::lang_unspecified);

    if (OpenCLLangStd == LangStandard::lang_unspecified) {
      Diags.Report(diag::err_drv_invalid_value)
      << A->getAsString(Args) << A->getValue();
    }
    else
      LangStd = OpenCLLangStd;
  }

  CompilerInvocation::setLangDefaults(Opts, IK, LangStd);

  // We abuse '-f[no-]gnu-keywords' to force overriding all GNU-extension
  // keywords. This behavior is provided by GCC's poorly named '-fasm' flag,
  // while a subset (the non-C++ GNU keywords) is provided by GCC's
  // '-fgnu-keywords'. Clang conflates the two for simplicity under the single
  // name, as it doesn't seem a useful distinction.
  Opts.GNUKeywords = Args.hasFlag(OPT_fgnu_keywords, OPT_fno_gnu_keywords,
                                  Opts.GNUKeywords);

  if (Args.hasArg(OPT_fno_operator_names))
    Opts.CXXOperatorNames = 0;

  if (Args.hasArg(OPT_fcuda_is_device))
    Opts.CUDAIsDevice = 1;

  if (Args.hasArg(OPT_fcuda_allow_host_calls_from_host_device))
    Opts.CUDAAllowHostCallsFromHostDevice = 1;

  if (Args.hasArg(OPT_fcuda_disable_target_call_checks))
    Opts.CUDADisableTargetCallChecks = 1;

  if (Args.hasArg(OPT_fcuda_target_overloads))
    Opts.CUDATargetOverloads = 1;

  if (Opts.ObjC1) {
    if (Arg *arg = Args.getLastArg(OPT_fobjc_runtime_EQ)) {
      StringRef value = arg->getValue();
      if (Opts.ObjCRuntime.tryParse(value))
        Diags.Report(diag::err_drv_unknown_objc_runtime) << value;
    }

    if (Args.hasArg(OPT_fobjc_gc_only))
      Opts.setGC(LangOptions::GCOnly);
    else if (Args.hasArg(OPT_fobjc_gc))
      Opts.setGC(LangOptions::HybridGC);
    else if (Args.hasArg(OPT_fobjc_arc)) {
      Opts.ObjCAutoRefCount = 1;
      if (!Opts.ObjCRuntime.allowsARC())
        Diags.Report(diag::err_arc_unsupported_on_runtime);
    }

    // ObjCWeakRuntime tracks whether the runtime supports __weak, not
    // whether the feature is actually enabled.  This is predominantly
    // determined by -fobjc-runtime, but we allow it to be overridden
    // from the command line for testing purposes.
    if (Args.hasArg(OPT_fobjc_runtime_has_weak))
      Opts.ObjCWeakRuntime = 1;
    else
      Opts.ObjCWeakRuntime = Opts.ObjCRuntime.allowsWeak();

    // ObjCWeak determines whether __weak is actually enabled.
    // Note that we allow -fno-objc-weak to disable this even in ARC mode.
    if (auto weakArg = Args.getLastArg(OPT_fobjc_weak, OPT_fno_objc_weak)) {
      if (!weakArg->getOption().matches(OPT_fobjc_weak)) {
        assert(!Opts.ObjCWeak);
      } else if (Opts.getGC() != LangOptions::NonGC) {
        Diags.Report(diag::err_objc_weak_with_gc);
      } else if (!Opts.ObjCWeakRuntime) {
        Diags.Report(diag::err_objc_weak_unsupported);
      } else {
        Opts.ObjCWeak = 1;
      }
    } else if (Opts.ObjCAutoRefCount) {
      Opts.ObjCWeak = Opts.ObjCWeakRuntime;
    }

    if (Args.hasArg(OPT_fno_objc_infer_related_result_type))
      Opts.ObjCInferRelatedResultType = 0;

    if (Args.hasArg(OPT_fobjc_subscripting_legacy_runtime))
      Opts.ObjCSubscriptingLegacyRuntime =
        (Opts.ObjCRuntime.getKind() == ObjCRuntime::FragileMacOSX);
  }

  if (Args.hasArg(OPT_fgnu89_inline)) {
    if (Opts.CPlusPlus)
      Diags.Report(diag::err_drv_argument_not_allowed_with) << "-fgnu89-inline"
                                                            << "C++/ObjC++";
    else
      Opts.GNUInline = 1;
  }

  if (Args.hasArg(OPT_fapple_kext)) {
    if (!Opts.CPlusPlus)
      Diags.Report(diag::warn_c_kext);
    else
      Opts.AppleKext = 1;
  }

  if (Args.hasArg(OPT_print_ivar_layout))
    Opts.ObjCGCBitmapPrint = 1;
  if (Args.hasArg(OPT_fno_constant_cfstrings))
    Opts.NoConstantCFStrings = 1;

  if (Args.hasArg(OPT_faltivec))
    Opts.AltiVec = 1;

  if (Args.hasArg(OPT_fzvector))
    Opts.ZVector = 1;

  if (Args.hasArg(OPT_pthread))
    Opts.POSIXThreads = 1;

  // The value-visibility mode defaults to "default".
  if (Arg *visOpt = Args.getLastArg(OPT_fvisibility)) {
    Opts.setValueVisibilityMode(parseVisibility(visOpt, Args, Diags));
  } else {
    Opts.setValueVisibilityMode(DefaultVisibility);
  }

  // The type-visibility mode defaults to the value-visibility mode.
  if (Arg *typeVisOpt = Args.getLastArg(OPT_ftype_visibility)) {
    Opts.setTypeVisibilityMode(parseVisibility(typeVisOpt, Args, Diags));
  } else {
    Opts.setTypeVisibilityMode(Opts.getValueVisibilityMode());
  }

  if (Args.hasArg(OPT_fvisibility_inlines_hidden))
    Opts.InlineVisibilityHidden = 1;

  if (Args.hasArg(OPT_ftrapv)) {
    Opts.setSignedOverflowBehavior(LangOptions::SOB_Trapping);
    // Set the handler, if one is specified.
    Opts.OverflowHandler =
        Args.getLastArgValue(OPT_ftrapv_handler);
  }
  else if (Args.hasArg(OPT_fwrapv))
    Opts.setSignedOverflowBehavior(LangOptions::SOB_Defined);

  Opts.MSVCCompat = Args.hasArg(OPT_fms_compatibility);
  Opts.MicrosoftExt = Opts.MSVCCompat || Args.hasArg(OPT_fms_extensions);
  Opts.AsmBlocks = Args.hasArg(OPT_fasm_blocks) || Opts.MicrosoftExt;
  Opts.MSCompatibilityVersion = 0;
  if (const Arg *A = Args.getLastArg(OPT_fms_compatibility_version)) {
    VersionTuple VT;
    if (VT.tryParse(A->getValue()))
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args)
                                                << A->getValue();
    Opts.MSCompatibilityVersion = VT.getMajor() * 10000000 +
                                  VT.getMinor().getValueOr(0) * 100000 +
                                  VT.getSubminor().getValueOr(0);
  }

  // Mimicing gcc's behavior, trigraphs are only enabled if -trigraphs
  // is specified, or -std is set to a conforming mode.
  // Trigraphs are disabled by default in c++1z onwards.
  Opts.Trigraphs = !Opts.GNUMode && !Opts.MSVCCompat && !Opts.CPlusPlus1z;
  Opts.Trigraphs =
      Args.hasFlag(OPT_ftrigraphs, OPT_fno_trigraphs, Opts.Trigraphs);

  Opts.DollarIdents = Args.hasFlag(OPT_fdollars_in_identifiers,
                                   OPT_fno_dollars_in_identifiers,
                                   Opts.DollarIdents);
  Opts.PascalStrings = Args.hasArg(OPT_fpascal_strings);
  Opts.VtorDispMode = getLastArgIntValue(Args, OPT_vtordisp_mode_EQ, 1, Diags);
  Opts.Borland = Args.hasArg(OPT_fborland_extensions);
  Opts.WritableStrings = Args.hasArg(OPT_fwritable_strings);
  Opts.ConstStrings = Args.hasFlag(OPT_fconst_strings, OPT_fno_const_strings,
                                   Opts.ConstStrings);
  if (Args.hasArg(OPT_fno_lax_vector_conversions))
    Opts.LaxVectorConversions = 0;
  if (Args.hasArg(OPT_fno_threadsafe_statics))
    Opts.ThreadsafeStatics = 0;
  Opts.Exceptions = Args.hasArg(OPT_fexceptions);
  Opts.ObjCExceptions = Args.hasArg(OPT_fobjc_exceptions);
  Opts.CXXExceptions = Args.hasArg(OPT_fcxx_exceptions);
  Opts.SjLjExceptions = Args.hasArg(OPT_fsjlj_exceptions);
  Opts.TraditionalCPP = Args.hasArg(OPT_traditional_cpp);

  Opts.RTTI = !Args.hasArg(OPT_fno_rtti);
  Opts.RTTIData = Opts.RTTI && !Args.hasArg(OPT_fno_rtti_data);
  Opts.Blocks = Args.hasArg(OPT_fblocks);
  Opts.BlocksRuntimeOptional = Args.hasArg(OPT_fblocks_runtime_optional);
  Opts.Coroutines = Args.hasArg(OPT_fcoroutines);
  Opts.Modules = Args.hasArg(OPT_fmodules);
  Opts.ModulesStrictDeclUse = Args.hasArg(OPT_fmodules_strict_decluse);
  Opts.ModulesDeclUse =
      Args.hasArg(OPT_fmodules_decluse) || Opts.ModulesStrictDeclUse;
  Opts.ModulesLocalVisibility =
      Args.hasArg(OPT_fmodules_local_submodule_visibility);
  Opts.ModulesSearchAll = Opts.Modules &&
    !Args.hasArg(OPT_fno_modules_search_all) &&
    Args.hasArg(OPT_fmodules_search_all);
  Opts.ModulesErrorRecovery = !Args.hasArg(OPT_fno_modules_error_recovery);
  Opts.ImplicitModules = !Args.hasArg(OPT_fno_implicit_modules);
  Opts.CharIsSigned = Opts.OpenCL || !Args.hasArg(OPT_fno_signed_char);
  Opts.WChar = Opts.CPlusPlus && !Args.hasArg(OPT_fno_wchar);
  Opts.ShortWChar = Args.hasFlag(OPT_fshort_wchar, OPT_fno_short_wchar, false);
  Opts.ShortEnums = Args.hasArg(OPT_fshort_enums);
  Opts.Freestanding = Args.hasArg(OPT_ffreestanding);
  Opts.NoBuiltin = Args.hasArg(OPT_fno_builtin) || Opts.Freestanding;
  if (!Opts.NoBuiltin)
    getAllNoBuiltinFuncValues(Args, Opts.NoBuiltinFuncs);
  Opts.NoMathBuiltin = Args.hasArg(OPT_fno_math_builtin);
  Opts.AssumeSaneOperatorNew = !Args.hasArg(OPT_fno_assume_sane_operator_new);
  Opts.SizedDeallocation = Args.hasArg(OPT_fsized_deallocation);
  Opts.ConceptsTS = Args.hasArg(OPT_fconcepts_ts);
  Opts.HeinousExtensions = Args.hasArg(OPT_fheinous_gnu_extensions);
  Opts.AccessControl = !Args.hasArg(OPT_fno_access_control);
  Opts.ElideConstructors = !Args.hasArg(OPT_fno_elide_constructors);
  Opts.MathErrno = !Opts.OpenCL && Args.hasArg(OPT_fmath_errno);
  Opts.InstantiationDepth =
      getLastArgIntValue(Args, OPT_ftemplate_depth, 256, Diags);
  Opts.ArrowDepth =
      getLastArgIntValue(Args, OPT_foperator_arrow_depth, 256, Diags);
  Opts.ConstexprCallDepth =
      getLastArgIntValue(Args, OPT_fconstexpr_depth, 512, Diags);
  Opts.ConstexprStepLimit =
      getLastArgIntValue(Args, OPT_fconstexpr_steps, 1048576, Diags);
  Opts.BracketDepth = getLastArgIntValue(Args, OPT_fbracket_depth, 256, Diags);
  Opts.DelayedTemplateParsing = Args.hasArg(OPT_fdelayed_template_parsing);
  Opts.NumLargeByValueCopy =
      getLastArgIntValue(Args, OPT_Wlarge_by_value_copy_EQ, 0, Diags);
  Opts.MSBitfields = Args.hasArg(OPT_mms_bitfields);
  Opts.ObjCConstantStringClass =
    Args.getLastArgValue(OPT_fconstant_string_class);
  Opts.ObjCDefaultSynthProperties =
    !Args.hasArg(OPT_disable_objc_default_synthesize_properties);
  Opts.EncodeExtendedBlockSig =
    Args.hasArg(OPT_fencode_extended_block_signature);
  Opts.EmitAllDecls = Args.hasArg(OPT_femit_all_decls);
  Opts.PackStruct = getLastArgIntValue(Args, OPT_fpack_struct_EQ, 0, Diags);
  Opts.MaxTypeAlign = getLastArgIntValue(Args, OPT_fmax_type_align_EQ, 0, Diags);
  Opts.PICLevel = getLastArgIntValue(Args, OPT_pic_level, 0, Diags);
  Opts.PIELevel = getLastArgIntValue(Args, OPT_pie_level, 0, Diags);
  Opts.Static = Args.hasArg(OPT_static_define);
  Opts.DumpRecordLayoutsSimple = Args.hasArg(OPT_fdump_record_layouts_simple);
  Opts.DumpRecordLayouts = Opts.DumpRecordLayoutsSimple
                        || Args.hasArg(OPT_fdump_record_layouts);
  Opts.DumpVTableLayouts = Args.hasArg(OPT_fdump_vtable_layouts);
  Opts.SpellChecking = !Args.hasArg(OPT_fno_spell_checking);
  Opts.NoBitFieldTypeAlign = Args.hasArg(OPT_fno_bitfield_type_align);
  Opts.SinglePrecisionConstants = Args.hasArg(OPT_cl_single_precision_constant);
  Opts.FastRelaxedMath = Args.hasArg(OPT_cl_fast_relaxed_math);
  Opts.MRTD = Args.hasArg(OPT_mrtd);
  Opts.HexagonQdsp6Compat = Args.hasArg(OPT_mqdsp6_compat);
  Opts.FakeAddressSpaceMap = Args.hasArg(OPT_ffake_address_space_map);
  Opts.ParseUnknownAnytype = Args.hasArg(OPT_funknown_anytype);
  Opts.DebuggerSupport = Args.hasArg(OPT_fdebugger_support);
  Opts.DebuggerCastResultToId = Args.hasArg(OPT_fdebugger_cast_result_to_id);
  Opts.DebuggerObjCLiteral = Args.hasArg(OPT_fdebugger_objc_literal);
  Opts.ApplePragmaPack = Args.hasArg(OPT_fapple_pragma_pack);
  Opts.CurrentModule = Args.getLastArgValue(OPT_fmodule_name);
  Opts.AppExt = Args.hasArg(OPT_fapplication_extension);
  Opts.ImplementationOfModule =
      Args.getLastArgValue(OPT_fmodule_implementation_of);
  Opts.ModuleFeatures = Args.getAllArgValues(OPT_fmodule_feature);
  std::sort(Opts.ModuleFeatures.begin(), Opts.ModuleFeatures.end());
  Opts.NativeHalfType |= Args.hasArg(OPT_fnative_half_type);
  Opts.HalfArgsAndReturns = Args.hasArg(OPT_fallow_half_arguments_and_returns);
  Opts.APINotes = Args.hasArg(OPT_fapinotes);
  Opts.GNUAsm = !Args.hasArg(OPT_fno_gnu_inline_asm);

  // __declspec is enabled by default for the PS4 by the driver, and also
  // enabled for Microsoft Extensions or Borland Extensions, here.
  //
  // FIXME: __declspec is also currently enabled for CUDA, but isn't really a
  // CUDA extension, however it is required for supporting cuda_builtin_vars.h,
  // which uses __declspec(property). Once that has been rewritten in terms of
  // something more generic, remove the Opts.CUDA term here.
  Opts.DeclSpecKeyword =
      Args.hasFlag(OPT_fdeclspec, OPT_fno_declspec,
                   (Opts.MicrosoftExt || Opts.Borland || Opts.CUDA));

  if (!Opts.CurrentModule.empty() && !Opts.ImplementationOfModule.empty() &&
      Opts.CurrentModule != Opts.ImplementationOfModule) {
    Diags.Report(diag::err_conflicting_module_names)
        << Opts.CurrentModule << Opts.ImplementationOfModule;
  }

  // For now, we only support local submodule visibility in C++ (because we
  // heavily depend on the ODR for merging redefinitions).
  if (Opts.ModulesLocalVisibility && !Opts.CPlusPlus)
    Diags.Report(diag::err_drv_argument_not_allowed_with)
        << "-fmodules-local-submodule-visibility" << "C";

  if (Arg *A = Args.getLastArg(OPT_faddress_space_map_mangling_EQ)) {
    switch (llvm::StringSwitch<unsigned>(A->getValue())
      .Case("target", LangOptions::ASMM_Target)
      .Case("no", LangOptions::ASMM_Off)
      .Case("yes", LangOptions::ASMM_On)
      .Default(255)) {
    default:
      Diags.Report(diag::err_drv_invalid_value)
        << "-faddress-space-map-mangling=" << A->getValue();
      break;
    case LangOptions::ASMM_Target:
      Opts.setAddressSpaceMapMangling(LangOptions::ASMM_Target);
      break;
    case LangOptions::ASMM_On:
      Opts.setAddressSpaceMapMangling(LangOptions::ASMM_On);
      break;
    case LangOptions::ASMM_Off:
      Opts.setAddressSpaceMapMangling(LangOptions::ASMM_Off);
      break;
    }
  }

  if (Arg *A = Args.getLastArg(OPT_fms_memptr_rep_EQ)) {
    LangOptions::PragmaMSPointersToMembersKind InheritanceModel =
        llvm::StringSwitch<LangOptions::PragmaMSPointersToMembersKind>(
            A->getValue())
            .Case("single",
                  LangOptions::PPTMK_FullGeneralitySingleInheritance)
            .Case("multiple",
                  LangOptions::PPTMK_FullGeneralityMultipleInheritance)
            .Case("virtual",
                  LangOptions::PPTMK_FullGeneralityVirtualInheritance)
            .Default(LangOptions::PPTMK_BestCase);
    if (InheritanceModel == LangOptions::PPTMK_BestCase)
      Diags.Report(diag::err_drv_invalid_value)
          << "-fms-memptr-rep=" << A->getValue();

    Opts.setMSPointerToMemberRepresentationMethod(InheritanceModel);
  }

  // Check if -fopenmp is specified.
  Opts.OpenMP = Args.hasArg(options::OPT_fopenmp);
  Opts.OpenMPUseTLS =
      Opts.OpenMP && !Args.hasArg(options::OPT_fnoopenmp_use_tls);
  Opts.OpenMPIsDevice =
      Opts.OpenMP && Args.hasArg(options::OPT_fopenmp_is_device);

  // Get the OpenMP target triples if any.
  if (Arg *A = Args.getLastArg(options::OPT_omptargets_EQ)) {

    for (unsigned i = 0; i < A->getNumValues(); ++i) {
      llvm::Triple TT(A->getValue(i));

      if (TT.getArch() == llvm::Triple::UnknownArch)
        Diags.Report(clang::diag::err_drv_invalid_omp_target) << A->getValue(i);
      else
        Opts.OMPTargetTriples.push_back(TT);
    }
  }

  // Get OpenMP host file path if any and report if a non existent file is
  // found
  if (Arg *A = Args.getLastArg(options::OPT_omp_host_ir_file_path)) {
    Opts.OMPHostIRFile = A->getValue();
    if (!llvm::sys::fs::exists(Opts.OMPHostIRFile))
      Diags.Report(clang::diag::err_drv_omp_host_ir_file_not_found)
          << Opts.OMPHostIRFile;
  }

  // Record whether the __DEPRECATED define was requested.
  Opts.Deprecated = Args.hasFlag(OPT_fdeprecated_macro,
                                 OPT_fno_deprecated_macro,
                                 Opts.Deprecated);

  // FIXME: Eliminate this dependency.
  unsigned Opt = getOptimizationLevel(Args, IK, Diags),
       OptSize = getOptimizationLevelSize(Args);
  Opts.Optimize = Opt != 0;
  Opts.OptimizeSize = OptSize != 0;

  // This is the __NO_INLINE__ define, which just depends on things like the
  // optimization level and -fno-inline, not actually whether the backend has
  // inlining enabled.
  Opts.NoInlineDefine = !Opt || Args.hasArg(OPT_fno_inline);

  Opts.FastMath = Args.hasArg(OPT_ffast_math) ||
      Args.hasArg(OPT_cl_fast_relaxed_math);
  Opts.FiniteMathOnly = Args.hasArg(OPT_ffinite_math_only) ||
      Args.hasArg(OPT_cl_finite_math_only) ||
      Args.hasArg(OPT_cl_fast_relaxed_math);
  Opts.UnsafeFPMath = Args.hasArg(OPT_menable_unsafe_fp_math) ||
                      Args.hasArg(OPT_cl_unsafe_math_optimizations) ||
                      Args.hasArg(OPT_cl_fast_relaxed_math);

  Opts.RetainCommentsFromSystemHeaders =
      Args.hasArg(OPT_fretain_comments_from_system_headers);

  unsigned SSP = getLastArgIntValue(Args, OPT_stack_protector, 0, Diags);
  switch (SSP) {
  default:
    Diags.Report(diag::err_drv_invalid_value)
      << Args.getLastArg(OPT_stack_protector)->getAsString(Args) << SSP;
    break;
  case 0: Opts.setStackProtector(LangOptions::SSPOff); break;
  case 1: Opts.setStackProtector(LangOptions::SSPOn);  break;
  case 2: Opts.setStackProtector(LangOptions::SSPStrong); break;
  case 3: Opts.setStackProtector(LangOptions::SSPReq); break;
  }

  // Parse -fsanitize= arguments.
  parseSanitizerKinds("-fsanitize=", Args.getAllArgValues(OPT_fsanitize_EQ),
                      Diags, Opts.Sanitize);
  // -fsanitize-address-field-padding=N has to be a LangOpt, parse it here.
  Opts.SanitizeAddressFieldPadding =
      getLastArgIntValue(Args, OPT_fsanitize_address_field_padding, 0, Diags);
  Opts.SanitizerBlacklistFiles = Args.getAllArgValues(OPT_fsanitize_blacklist);
}

static void ParsePreprocessorArgs(PreprocessorOptions &Opts, ArgList &Args,
                                  FileManager &FileMgr,
                                  DiagnosticsEngine &Diags) {
  using namespace options;
  Opts.ImplicitPCHInclude = Args.getLastArgValue(OPT_include_pch);
  Opts.ImplicitPTHInclude = Args.getLastArgValue(OPT_include_pth);
  if (const Arg *A = Args.getLastArg(OPT_token_cache))
      Opts.TokenCache = A->getValue();
  else
    Opts.TokenCache = Opts.ImplicitPTHInclude;
  Opts.UsePredefines = !Args.hasArg(OPT_undef);
  Opts.DetailedRecord = Args.hasArg(OPT_detailed_preprocessing_record);
  Opts.DisablePCHValidation = Args.hasArg(OPT_fno_validate_pch);

  Opts.DumpDeserializedPCHDecls = Args.hasArg(OPT_dump_deserialized_pch_decls);
  for (const Arg *A : Args.filtered(OPT_error_on_deserialized_pch_decl))
    Opts.DeserializedPCHDeclsToErrorOn.insert(A->getValue());

  if (const Arg *A = Args.getLastArg(OPT_preamble_bytes_EQ)) {
    StringRef Value(A->getValue());
    size_t Comma = Value.find(',');
    unsigned Bytes = 0;
    unsigned EndOfLine = 0;

    if (Comma == StringRef::npos ||
        Value.substr(0, Comma).getAsInteger(10, Bytes) ||
        Value.substr(Comma + 1).getAsInteger(10, EndOfLine))
      Diags.Report(diag::err_drv_preamble_format);
    else {
      Opts.PrecompiledPreambleBytes.first = Bytes;
      Opts.PrecompiledPreambleBytes.second = (EndOfLine != 0);
    }
  }

  // Add macros from the command line.
  for (const Arg *A : Args.filtered(OPT_D, OPT_U)) {
    if (A->getOption().matches(OPT_D))
      Opts.addMacroDef(A->getValue());
    else
      Opts.addMacroUndef(A->getValue());
  }

  Opts.MacroIncludes = Args.getAllArgValues(OPT_imacros);

  // Add the ordered list of -includes.
  for (const Arg *A : Args.filtered(OPT_include))
    Opts.Includes.emplace_back(A->getValue());

  for (const Arg *A : Args.filtered(OPT_chain_include))
    Opts.ChainedIncludes.emplace_back(A->getValue());

  // Include 'altivec.h' if -faltivec option present
  if (Args.hasArg(OPT_faltivec))
    Opts.Includes.emplace_back("altivec.h");

  for (const Arg *A : Args.filtered(OPT_remap_file)) {
    std::pair<StringRef, StringRef> Split = StringRef(A->getValue()).split(';');

    if (Split.second.empty()) {
      Diags.Report(diag::err_drv_invalid_remap_file) << A->getAsString(Args);
      continue;
    }

    Opts.addRemappedFile(Split.first, Split.second);
  }

  if (Arg *A = Args.getLastArg(OPT_fobjc_arc_cxxlib_EQ)) {
    StringRef Name = A->getValue();
    unsigned Library = llvm::StringSwitch<unsigned>(Name)
      .Case("libc++", ARCXX_libcxx)
      .Case("libstdc++", ARCXX_libstdcxx)
      .Case("none", ARCXX_nolib)
      .Default(~0U);
    if (Library == ~0U)
      Diags.Report(diag::err_drv_invalid_value) << A->getAsString(Args) << Name;
    else
      Opts.ObjCXXARCStandardLibrary = (ObjCXXARCStandardLibraryKind)Library;
  }
}

static void ParsePreprocessorOutputArgs(PreprocessorOutputOptions &Opts,
                                        ArgList &Args,
                                        frontend::ActionKind Action) {
  using namespace options;

  switch (Action) {
  case frontend::ASTDeclList:
  case frontend::ASTDump:
  case frontend::ASTPrint:
  case frontend::ASTView:
  case frontend::EmitAssembly:
  case frontend::EmitBC:
  case frontend::EmitHTML:
  case frontend::EmitLLVM:
  case frontend::EmitLLVMOnly:
  case frontend::EmitCodeGenOnly:
  case frontend::EmitObj:
  case frontend::FixIt:
  case frontend::GenerateModule:
  case frontend::GeneratePCH:
  case frontend::GeneratePTH:
  case frontend::ParseSyntaxOnly:
  case frontend::ModuleFileInfo:
  case frontend::VerifyPCH:
  case frontend::PluginAction:
  case frontend::PrintDeclContext:
  case frontend::RewriteObjC:
  case frontend::RewriteTest:
  case frontend::RunAnalysis:
  case frontend::MigrateSource:
    Opts.ShowCPP = 0;
    break;

  case frontend::DumpRawTokens:
  case frontend::DumpTokens:
  case frontend::InitOnly:
  case frontend::PrintPreamble:
  case frontend::PrintPreprocessedInput:
  case frontend::RewriteMacros:
  case frontend::RunPreprocessorOnly:
    Opts.ShowCPP = !Args.hasArg(OPT_dM);
    break;
  }

  Opts.ShowComments = Args.hasArg(OPT_C);
  Opts.ShowLineMarkers = !Args.hasArg(OPT_P);
  Opts.ShowMacroComments = Args.hasArg(OPT_CC);
  Opts.ShowMacros = Args.hasArg(OPT_dM) || Args.hasArg(OPT_dD);
  Opts.RewriteIncludes = Args.hasArg(OPT_frewrite_includes);
  Opts.UseLineDirectives = Args.hasArg(OPT_fuse_line_directives);
}

static void ParseTargetArgs(TargetOptions &Opts, ArgList &Args) {
  using namespace options;
  Opts.ABI = Args.getLastArgValue(OPT_target_abi);
  Opts.CPU = Args.getLastArgValue(OPT_target_cpu);
  Opts.FPMath = Args.getLastArgValue(OPT_mfpmath);
  Opts.FeaturesAsWritten = Args.getAllArgValues(OPT_target_feature);
  Opts.LinkerVersion = Args.getLastArgValue(OPT_target_linker_version);
  Opts.Triple = llvm::Triple::normalize(Args.getLastArgValue(OPT_triple));
  Opts.Reciprocals = Args.getAllArgValues(OPT_mrecip_EQ);
  // Use the default target triple if unspecified.
  if (Opts.Triple.empty())
    Opts.Triple = llvm::sys::getDefaultTargetTriple();
}

bool CompilerInvocation::CreateFromArgs(CompilerInvocation &Res,
                                        const char *const *ArgBegin,
                                        const char *const *ArgEnd,
                                        DiagnosticsEngine &Diags) {
  bool Success = true;

  // Parse the arguments.
  std::unique_ptr<OptTable> Opts(createDriverOptTable());
  const unsigned IncludedFlagsBitmask = options::CC1Option;
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList Args =
      Opts->ParseArgs(llvm::makeArrayRef(ArgBegin, ArgEnd), MissingArgIndex,
                      MissingArgCount, IncludedFlagsBitmask);

  // Check for missing argument error.
  if (MissingArgCount) {
    Diags.Report(diag::err_drv_missing_argument)
        << Args.getArgString(MissingArgIndex) << MissingArgCount;
    Success = false;
  }

  // Issue errors on unknown arguments.
  for (const Arg *A : Args.filtered(OPT_UNKNOWN)) {
    Diags.Report(diag::err_drv_unknown_argument) << A->getAsString(Args);
    Success = false;
  }

  Success &= ParseAnalyzerArgs(*Res.getAnalyzerOpts(), Args, Diags);
  Success &= ParseMigratorArgs(Res.getMigratorOpts(), Args);
  ParseDependencyOutputArgs(Res.getDependencyOutputOpts(), Args);
  Success &= ParseDiagnosticArgs(Res.getDiagnosticOpts(), Args, &Diags,
                                 false /*DefaultDiagColor*/);
  ParseCommentArgs(Res.getLangOpts()->CommentOpts, Args);
  ParseFileSystemArgs(Res.getFileSystemOpts(), Args);
  // FIXME: We shouldn't have to pass the DashX option around here
  InputKind DashX = ParseFrontendArgs(Res.getFrontendOpts(), Args, Diags);
  ParseTargetArgs(Res.getTargetOpts(), Args);
  Success &= ParseCodeGenArgs(Res.getCodeGenOpts(), Args, DashX, Diags,
                              Res.getTargetOpts());
  ParseHeaderSearchArgs(Res.getHeaderSearchOpts(), Args);
  ParseAPINotesArgs(Res.getAPINotesOpts(), Args);

  if (DashX == IK_AST || DashX == IK_LLVM_IR) {
    // ObjCAAutoRefCount and Sanitize LangOpts are used to setup the
    // PassManager in BackendUtil.cpp. They need to be initializd no matter
    // what the input type is.
    if (Args.hasArg(OPT_fobjc_arc))
      Res.getLangOpts()->ObjCAutoRefCount = 1;
    parseSanitizerKinds("-fsanitize=", Args.getAllArgValues(OPT_fsanitize_EQ),
                        Diags, Res.getLangOpts()->Sanitize);
  } else {
    // Other LangOpts are only initialzed when the input is not AST or LLVM IR.
    ParseLangArgs(*Res.getLangOpts(), Args, DashX, Diags);
    if (Res.getFrontendOpts().ProgramAction == frontend::RewriteObjC)
      Res.getLangOpts()->ObjCExceptions = 1;

    // -fapinotes requires -fapinotes-cache-path=<directory>.
    if (Res.getLangOpts()->APINotes &&
        Res.getFileSystemOpts().APINotesCachePath.empty()) {
      Diags.Report(diag::err_no_apinotes_cache_path);
      Success = false;
    }
  }

  // FIXME: Override value name discarding when asan or msan is used because the
  // backend passes depend on the name of the alloca in order to print out
  // names.
  Res.getCodeGenOpts().DiscardValueNames &=
      !Res.getLangOpts()->Sanitize.has(SanitizerKind::Address) &&
      !Res.getLangOpts()->Sanitize.has(SanitizerKind::Memory);

  // FIXME: ParsePreprocessorArgs uses the FileManager to read the contents of
  // PCH file and find the original header name. Remove the need to do that in
  // ParsePreprocessorArgs and remove the FileManager
  // parameters from the function and the "FileManager.h" #include.
  FileManager FileMgr(Res.getFileSystemOpts());
  ParsePreprocessorArgs(Res.getPreprocessorOpts(), Args, FileMgr, Diags);
  ParsePreprocessorOutputArgs(Res.getPreprocessorOutputOpts(), Args,
                              Res.getFrontendOpts().ProgramAction);
  return Success;
}

namespace {

  class ModuleSignature {
    SmallVector<uint64_t, 16> Data;
    unsigned CurBit;
    uint64_t CurValue;

  public:
    ModuleSignature() : CurBit(0), CurValue(0) { }

    void add(uint64_t Value, unsigned Bits);
    void add(StringRef Value);
    void flush();

    llvm::APInt getAsInteger() const;
  };
}

void ModuleSignature::add(uint64_t Value, unsigned int NumBits) {
  CurValue |= Value << CurBit;
  if (CurBit + NumBits < 64) {
    CurBit += NumBits;
    return;
  }

  // Add the current word.
  Data.push_back(CurValue);

  if (CurBit)
    CurValue = Value >> (64-CurBit);
  else
    CurValue = 0;
  CurBit = (CurBit+NumBits) & 63;
}

void ModuleSignature::flush() {
  if (CurBit == 0)
    return;

  Data.push_back(CurValue);
  CurBit = 0;
  CurValue = 0;
}

void ModuleSignature::add(StringRef Value) {
  for (auto &c : Value)
    add(c, 8);
}

llvm::APInt ModuleSignature::getAsInteger() const {
  return llvm::APInt(Data.size() * 64, Data);
}

std::string CompilerInvocation::getModuleHash() const {
  // Note: For QoI reasons, the things we use as a hash here should all be
  // dumped via the -module-info flag.
  using llvm::hash_code;
  using llvm::hash_value;
  using llvm::hash_combine;

  // Start the signature with the compiler version.
  // FIXME: We'd rather use something more cryptographically sound than
  // CityHash, but this will do for now.
  hash_code code = hash_value(getClangFullRepositoryVersion());

  // Extend the signature with the language options
#define LANGOPT(Name, Bits, Default, Description) \
   code = hash_combine(code, LangOpts->Name);
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
  code = hash_combine(code, static_cast<unsigned>(LangOpts->get##Name()));
#define BENIGN_LANGOPT(Name, Bits, Default, Description)
#define BENIGN_ENUM_LANGOPT(Name, Type, Bits, Default, Description)
#include "clang/Basic/LangOptions.def"

  for (StringRef Feature : LangOpts->ModuleFeatures)
    code = hash_combine(code, Feature);

  // Extend the signature with the target options.
  code = hash_combine(code, TargetOpts->Triple, TargetOpts->CPU,
                      TargetOpts->ABI);
  for (unsigned i = 0, n = TargetOpts->FeaturesAsWritten.size(); i != n; ++i)
    code = hash_combine(code, TargetOpts->FeaturesAsWritten[i]);

  // Extend the signature with preprocessor options.
  const PreprocessorOptions &ppOpts = getPreprocessorOpts();
  const HeaderSearchOptions &hsOpts = getHeaderSearchOpts();
  code = hash_combine(code, ppOpts.UsePredefines, ppOpts.DetailedRecord);

  for (std::vector<std::pair<std::string, bool/*isUndef*/>>::const_iterator
            I = getPreprocessorOpts().Macros.begin(),
         IEnd = getPreprocessorOpts().Macros.end();
       I != IEnd; ++I) {
    // If we're supposed to ignore this macro for the purposes of modules,
    // don't put it into the hash.
    if (!hsOpts.ModulesIgnoreMacros.empty()) {
      // Check whether we're ignoring this macro.
      StringRef MacroDef = I->first;
      if (hsOpts.ModulesIgnoreMacros.count(MacroDef.split('=').first))
        continue;
    }

    code = hash_combine(code, I->first, I->second);
  }

  // Extend the signature with the sysroot and other header search options.
  code = hash_combine(code, hsOpts.Sysroot,
                      hsOpts.ModuleFormat,
                      hsOpts.UseDebugInfo,
                      hsOpts.UseBuiltinIncludes,
                      hsOpts.UseStandardSystemIncludes,
                      hsOpts.UseStandardCXXIncludes,
                      hsOpts.UseLibcxx,
                      hsOpts.ModulesValidateDiagnosticOptions);
  code = hash_combine(code, hsOpts.ResourceDir);

  // Extend the signature with the user build path.
  code = hash_combine(code, hsOpts.ModuleUserBuildPath);

  // Extend the signature with the module file extensions.
  const FrontendOptions &frontendOpts = getFrontendOpts();
  for (auto ext : frontendOpts.ModuleFileExtensions) {
    code = ext->hashExtension(code);
  }

  // Darwin-specific hack: if we have a sysroot, use the contents and
  // modification time of
  //   $sysroot/System/Library/CoreServices/SystemVersion.plist
  // as part of the module hash.
  if (!hsOpts.Sysroot.empty()) {
    SmallString<128> systemVersionFile;
    systemVersionFile += hsOpts.Sysroot;
    llvm::sys::path::append(systemVersionFile, "System");
    llvm::sys::path::append(systemVersionFile, "Library");
    llvm::sys::path::append(systemVersionFile, "CoreServices");
    llvm::sys::path::append(systemVersionFile, "SystemVersion.plist");

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(systemVersionFile);
    if (buffer) {
      code = hash_combine(code, buffer.get()->getBuffer());

      struct stat statBuf;
      if (stat(systemVersionFile.c_str(), &statBuf) == 0)
        code = hash_combine(code, statBuf.st_mtime);
    }
  }

  return llvm::APInt(64, code).toString(36, /*Signed=*/false);
}

namespace clang {

template<typename IntTy>
static IntTy getLastArgIntValueImpl(const ArgList &Args, OptSpecifier Id,
                                    IntTy Default,
                                    DiagnosticsEngine *Diags) {
  IntTy Res = Default;
  if (Arg *A = Args.getLastArg(Id)) {
    if (StringRef(A->getValue()).getAsInteger(10, Res)) {
      if (Diags)
        Diags->Report(diag::err_drv_invalid_int_value) << A->getAsString(Args)
                                                       << A->getValue();
    }
  }
  return Res;
}


// Declared in clang/Frontend/Utils.h.
int getLastArgIntValue(const ArgList &Args, OptSpecifier Id, int Default,
                       DiagnosticsEngine *Diags) {
  return getLastArgIntValueImpl<int>(Args, Id, Default, Diags);
}

uint64_t getLastArgUInt64Value(const ArgList &Args, OptSpecifier Id,
                               uint64_t Default,
                               DiagnosticsEngine *Diags) {
  return getLastArgIntValueImpl<uint64_t>(Args, Id, Default, Diags);
}

void BuryPointer(const void *Ptr) {
  // This function may be called only a small fixed amount of times per each
  // invocation, otherwise we do actually have a leak which we want to report.
  // If this function is called more than kGraveYardMaxSize times, the pointers
  // will not be properly buried and a leak detector will report a leak, which
  // is what we want in such case.
  static const size_t kGraveYardMaxSize = 16;
  LLVM_ATTRIBUTE_UNUSED static const void *GraveYard[kGraveYardMaxSize];
  static std::atomic<unsigned> GraveYardSize;
  unsigned Idx = GraveYardSize++;
  if (Idx >= kGraveYardMaxSize)
    return;
  GraveYard[Idx] = Ptr;
}

IntrusiveRefCntPtr<vfs::FileSystem>
createVFSFromCompilerInvocation(const CompilerInvocation &CI,
                                DiagnosticsEngine &Diags) {
  if (CI.getHeaderSearchOpts().VFSOverlayFiles.empty())
    return vfs::getRealFileSystem();

  IntrusiveRefCntPtr<vfs::OverlayFileSystem>
    Overlay(new vfs::OverlayFileSystem(vfs::getRealFileSystem()));
  // earlier vfs files are on the bottom
  for (const std::string &File : CI.getHeaderSearchOpts().VFSOverlayFiles) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
        llvm::MemoryBuffer::getFile(File);
    if (!Buffer) {
      Diags.Report(diag::err_missing_vfs_overlay_file) << File;
      return IntrusiveRefCntPtr<vfs::FileSystem>();
    }

    IntrusiveRefCntPtr<vfs::FileSystem> FS = vfs::getVFSFromYAML(
        std::move(Buffer.get()), /*DiagHandler*/ nullptr, File);
    if (!FS.get()) {
      Diags.Report(diag::err_invalid_vfs_overlay) << File;
      return IntrusiveRefCntPtr<vfs::FileSystem>();
    }
    Overlay->pushOverlay(FS);
  }
  return Overlay;
}
} // end namespace clang
