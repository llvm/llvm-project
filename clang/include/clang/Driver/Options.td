//===--- Options.td - Options for clang -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by clang.
//
//===----------------------------------------------------------------------===//

// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"

/////////
// Flags

// The option is a "driver"-only option, and should not be forwarded to other
// tools via `-Xarch` options.
def NoXarchOption : OptionFlag;

// LinkerInput - The option is a linker input.
def LinkerInput : OptionFlag;

// NoArgumentUnused - Don't report argument unused warnings for this option; this
// is useful for options like -static or -dynamic which a user may always end up
// passing, even if the platform defaults to (or only supports) that option.
def NoArgumentUnused : OptionFlag;

// Unsupported - The option is unsupported, and the driver will reject command
// lines that use it.
def Unsupported : OptionFlag;

// Ignored - The option is unsupported, and the driver will silently ignore it.
def Ignored : OptionFlag;

// CoreOption - This is considered a "core" Clang option, available in both
// clang and clang-cl modes.
def CoreOption : OptionFlag;

// CLOption - This is a cl.exe compatibility option. Options with this flag
// are made available when the driver is running in CL compatibility mode.
def CLOption : OptionFlag;

// CC1Option - This option should be accepted by clang -cc1.
def CC1Option : OptionFlag;

// CC1AsOption - This option should be accepted by clang -cc1as.
def CC1AsOption : OptionFlag;

// DXCOption - This is a dxc.exe compatibility option. Options with this flag
// are made available when the driver is running in DXC compatibility mode.
def DXCOption : OptionFlag;

// CLDXCOption - This is a cl.exe/dxc.exe compatibility option. Options with this flag
// are made available when the driver is running in CL/DXC compatibility mode.
def CLDXCOption : OptionFlag;

// NoDriverOption - This option should not be accepted by the driver.
def NoDriverOption : OptionFlag;

// If an option affects linking, but has a primary group (so Link_Group cannot
// be used), add this flag.
def LinkOption : OptionFlag;

// FlangOption - This is considered a "core" Flang option, available in
// flang mode.
def FlangOption : OptionFlag;

// FlangOnlyOption - This option should only be used by Flang (i.e. it is not
// available for Clang)
def FlangOnlyOption : OptionFlag;

// FC1Option - This option should be accepted by flang -fc1.
def FC1Option : OptionFlag;

// A short name to show in documentation. The name will be interpreted as rST.
class DocName<string name> { string DocName = name; }

// A brief description to show in documentation, interpreted as rST.
class DocBrief<code descr> { code DocBrief = descr; }

// Indicates that this group should be flattened into its parent when generating
// documentation.
class DocFlatten { bit DocFlatten = 1; }

// Indicates that this warning is ignored, but accepted with a warning for
// GCC compatibility.
class IgnoredGCCCompat : Flags<[HelpHidden]> {}

/////////
// Groups

def Action_Group : OptionGroup<"<action group>">, DocName<"Actions">,
                   DocBrief<[{The action to perform on the input.}]>;

// Meta-group for options which are only used for compilation,
// and not linking etc.
def CompileOnly_Group : OptionGroup<"<CompileOnly group>">,
                        DocName<"Compilation flags">, DocBrief<[{
Flags controlling the behavior of Clang during compilation. These flags have
no effect during actions that do not perform compilation.}]>;

def Preprocessor_Group : OptionGroup<"<Preprocessor group>">,
                         Group<CompileOnly_Group>,
                         DocName<"Preprocessor flags">, DocBrief<[{
Flags controlling the behavior of the Clang preprocessor.}]>;

def IncludePath_Group : OptionGroup<"<I/i group>">, Group<Preprocessor_Group>,
                        DocName<"Include path management">,
                        DocBrief<[{
Flags controlling how ``#include``\s are resolved to files.}]>;

def I_Group : OptionGroup<"<I group>">, Group<IncludePath_Group>, DocFlatten;
def i_Group : OptionGroup<"<i group>">, Group<IncludePath_Group>, DocFlatten;
def clang_i_Group : OptionGroup<"<clang i group>">, Group<i_Group>, DocFlatten;

def M_Group : OptionGroup<"<M group>">, Group<Preprocessor_Group>,
              DocName<"Dependency file generation">, DocBrief<[{
Flags controlling generation of a dependency file for ``make``-like build
systems.}]>;

def d_Group : OptionGroup<"<d group>">, Group<Preprocessor_Group>,
              DocName<"Dumping preprocessor state">, DocBrief<[{
Flags allowing the state of the preprocessor to be dumped in various ways.}]>;

def Diag_Group : OptionGroup<"<W/R group>">, Group<CompileOnly_Group>,
                 DocName<"Diagnostic flags">, DocBrief<[{
Flags controlling which warnings, errors, and remarks Clang will generate.
See the :doc:`full list of warning and remark flags <DiagnosticsReference>`.}]>;

def R_Group : OptionGroup<"<R group>">, Group<Diag_Group>, DocFlatten;
def R_value_Group : OptionGroup<"<R (with value) group>">, Group<R_Group>,
                    DocFlatten;
def W_Group : OptionGroup<"<W group>">, Group<Diag_Group>, DocFlatten;
def W_value_Group : OptionGroup<"<W (with value) group>">, Group<W_Group>,
                    DocFlatten;

def f_Group : OptionGroup<"<f group>">, Group<CompileOnly_Group>,
              DocName<"Target-independent compilation options">;

def f_clang_Group : OptionGroup<"<f (clang-only) group>">,
                    Group<CompileOnly_Group>, DocFlatten;
def pedantic_Group : OptionGroup<"<pedantic group>">, Group<f_Group>,
                     DocFlatten;
def opencl_Group : OptionGroup<"<opencl group>">, Group<f_Group>,
                   DocName<"OpenCL flags">;

def sycl_Group : OptionGroup<"<SYCL group>">, Group<f_Group>,
                 DocName<"SYCL flags">;

def m_Group : OptionGroup<"<m group>">, Group<CompileOnly_Group>,
              DocName<"Target-dependent compilation options">;

// Feature groups - these take command line options that correspond directly to
// target specific features and can be translated directly from command line
// options.
def m_aarch64_Features_Group : OptionGroup<"<aarch64 features group>">,
                               Group<m_Group>, DocName<"AARCH64">;
def m_amdgpu_Features_Group : OptionGroup<"<amdgpu features group>">,
                              Group<m_Group>, DocName<"AMDGPU">;
def m_arm_Features_Group : OptionGroup<"<arm features group>">,
                           Group<m_Group>, DocName<"ARM">;
def m_hexagon_Features_Group : OptionGroup<"<hexagon features group>">,
                               Group<m_Group>, DocName<"Hexagon">;
// The features added by this group will not be added to target features.
// These are explicitly handled.
def m_hexagon_Features_HVX_Group : OptionGroup<"<hexagon features group>">,
                                   Group<m_Group>, DocName<"Hexagon">;
def m_m68k_Features_Group: OptionGroup<"<m68k features group>">,
                           Group<m_Group>, DocName<"M68k">;
def m_mips_Features_Group : OptionGroup<"<mips features group>">,
                            Group<m_Group>, DocName<"MIPS">;
def m_ppc_Features_Group : OptionGroup<"<ppc features group>">,
                           Group<m_Group>, DocName<"PowerPC">;
def m_wasm_Features_Group : OptionGroup<"<wasm features group>">,
                            Group<m_Group>, DocName<"WebAssembly">;
// The features added by this group will not be added to target features.
// These are explicitly handled.
def m_wasm_Features_Driver_Group : OptionGroup<"<wasm driver features group>">,
                                   Group<m_Group>, DocName<"WebAssembly Driver">;
def m_x86_Features_Group : OptionGroup<"<x86 features group>">,
                           Group<m_Group>, Flags<[CoreOption]>, DocName<"X86">;
def m_riscv_Features_Group : OptionGroup<"<riscv features group>">,
                             Group<m_Group>, DocName<"RISCV">;

def m_libc_Group : OptionGroup<"<m libc group>">, Group<m_mips_Features_Group>,
                   Flags<[HelpHidden]>;

def O_Group : OptionGroup<"<O group>">, Group<CompileOnly_Group>,
              DocName<"Optimization level">, DocBrief<[{
Flags controlling how much optimization should be performed.}]>;

def DebugInfo_Group : OptionGroup<"<g group>">, Group<CompileOnly_Group>,
                      DocName<"Debug information generation">, DocBrief<[{
Flags controlling how much and what kind of debug information should be
generated.}]>;

def g_Group : OptionGroup<"<g group>">, Group<DebugInfo_Group>,
              DocName<"Kind and level of debug information">;
def gN_Group : OptionGroup<"<gN group>">, Group<g_Group>,
               DocName<"Debug level">;
def ggdbN_Group : OptionGroup<"<ggdbN group>">, Group<gN_Group>, DocFlatten;
def gTune_Group : OptionGroup<"<gTune group>">, Group<g_Group>,
                  DocName<"Debugger to tune debug information for">;
def g_flags_Group : OptionGroup<"<g flags group>">, Group<DebugInfo_Group>,
                    DocName<"Debug information flags">;

def StaticAnalyzer_Group : OptionGroup<"<Static analyzer group>">,
                           DocName<"Static analyzer flags">, DocBrief<[{
Flags controlling the behavior of the Clang Static Analyzer.}]>;

// gfortran options that we recognize in the driver and pass along when
// invoking GCC to compile Fortran code.
def gfortran_Group : OptionGroup<"<gfortran group>">,
                     DocName<"Fortran compilation flags">, DocBrief<[{
Flags that will be passed onto the ``gfortran`` compiler when Clang is given
a Fortran input.}]>;

def Link_Group : OptionGroup<"<T/e/s/t/u group>">, DocName<"Linker flags">,
                 DocBrief<[{Flags that are passed on to the linker}]>;
def T_Group : OptionGroup<"<T group>">, Group<Link_Group>, DocFlatten;
def u_Group : OptionGroup<"<u group>">, Group<Link_Group>, DocFlatten;

def reserved_lib_Group : OptionGroup<"<reserved libs group>">,
                         Flags<[Unsupported]>;

// Temporary groups for clang options which we know we don't support,
// but don't want to verbosely warn the user about.
def clang_ignored_f_Group : OptionGroup<"<clang ignored f group>">,
  Group<f_Group>, Flags<[Ignored]>;
def clang_ignored_m_Group : OptionGroup<"<clang ignored m group>">,
  Group<m_Group>, Flags<[Ignored]>;

// Group for clang options in the process of deprecation.
// Please include the version that deprecated the flag as comment to allow
// easier garbage collection.
def clang_ignored_legacy_options_Group : OptionGroup<"<clang legacy flags>">,
  Group<f_Group>, Flags<[Ignored]>;

// Retired with clang-5.0
def : Flag<["-"], "fslp-vectorize-aggressive">, Group<clang_ignored_legacy_options_Group>;
def : Flag<["-"], "fno-slp-vectorize-aggressive">, Group<clang_ignored_legacy_options_Group>;

// Retired with clang-10.0. Previously controlled X86 MPX ISA.
def mmpx : Flag<["-"], "mmpx">, Group<clang_ignored_legacy_options_Group>;
def mno_mpx : Flag<["-"], "mno-mpx">, Group<clang_ignored_legacy_options_Group>;

// Group that ignores all gcc optimizations that won't be implemented
def clang_ignored_gcc_optimization_f_Group : OptionGroup<
  "<clang_ignored_gcc_optimization_f_Group>">, Group<f_Group>, Flags<[Ignored]>;

class DiagnosticOpts<string base>
  : KeyPathAndMacro<"DiagnosticOpts->", base, "DIAG_"> {}
class LangOpts<string base>
  : KeyPathAndMacro<"LangOpts->", base, "LANG_"> {}
class TargetOpts<string base>
  : KeyPathAndMacro<"TargetOpts->", base, "TARGET_"> {}
class FrontendOpts<string base>
  : KeyPathAndMacro<"FrontendOpts.", base, "FRONTEND_"> {}
class PreprocessorOutputOpts<string base>
  : KeyPathAndMacro<"PreprocessorOutputOpts.", base, "PREPROCESSOR_OUTPUT_"> {}
class DependencyOutputOpts<string base>
  : KeyPathAndMacro<"DependencyOutputOpts.", base, "DEPENDENCY_OUTPUT_"> {}
class CodeGenOpts<string base>
  : KeyPathAndMacro<"CodeGenOpts.", base, "CODEGEN_"> {}
class HeaderSearchOpts<string base>
  : KeyPathAndMacro<"HeaderSearchOpts->", base, "HEADER_SEARCH_"> {}
class PreprocessorOpts<string base>
  : KeyPathAndMacro<"PreprocessorOpts->", base, "PREPROCESSOR_"> {}
class FileSystemOpts<string base>
  : KeyPathAndMacro<"FileSystemOpts.", base, "FILE_SYSTEM_"> {}
class AnalyzerOpts<string base>
  : KeyPathAndMacro<"AnalyzerOpts->", base, "ANALYZER_"> {}
class MigratorOpts<string base>
  : KeyPathAndMacro<"MigratorOpts.", base, "MIGRATOR_"> {}

// A boolean option which is opt-in in CC1. The positive option exists in CC1 and
// Args.hasArg(OPT_ffoo) can be used to check that the flag is enabled.
// This is useful if the option is usually disabled.
// Use this only when the option cannot be declared via BoolFOption.
multiclass OptInCC1FFlag<string name, string pos_prefix, string neg_prefix="",
                      string help="", list<OptionFlag> flags=[]> {
  def f#NAME : Flag<["-"], "f"#name>, Flags<[CC1Option] # flags>,
               Group<f_Group>, HelpText<pos_prefix # help>;
  def fno_#NAME : Flag<["-"], "fno-"#name>, Flags<flags>,
                  Group<f_Group>, HelpText<neg_prefix # help>;
}

// A boolean option which is opt-out in CC1. The negative option exists in CC1 and
// Args.hasArg(OPT_fno_foo) can be used to check that the flag is disabled.
// Use this only when the option cannot be declared via BoolFOption.
multiclass OptOutCC1FFlag<string name, string pos_prefix, string neg_prefix,
                       string help="", list<OptionFlag> flags=[]> {
  def f#NAME : Flag<["-"], "f"#name>, Flags<flags>,
               Group<f_Group>, HelpText<pos_prefix # help>;
  def fno_#NAME : Flag<["-"], "fno-"#name>, Flags<[CC1Option] # flags>,
                  Group<f_Group>, HelpText<neg_prefix # help>;
}

// A boolean option which is opt-in in FC1. The positive option exists in FC1 and
// Args.hasArg(OPT_ffoo) can be used to check that the flag is enabled.
// This is useful if the option is usually disabled.
multiclass OptInFC1FFlag<string name, string pos_prefix, string neg_prefix="",
                      string help="", list<OptionFlag> flags=[]> {
  def f#NAME : Flag<["-"], "f"#name>, Flags<[FC1Option] # flags>,
               Group<f_Group>, HelpText<pos_prefix # help>;
  def fno_#NAME : Flag<["-"], "fno-"#name>, Flags<flags>,
                  Group<f_Group>, HelpText<neg_prefix # help>;
}

// A boolean option which is opt-out in FC1. The negative option exists in FC1 and
// Args.hasArg(OPT_fno_foo) can be used to check that the flag is disabled.
multiclass OptOutFC1FFlag<string name, string pos_prefix, string neg_prefix,
                       string help="", list<OptionFlag> flags=[]> {
  def f#NAME : Flag<["-"], "f"#name>, Flags<flags>,
               Group<f_Group>, HelpText<pos_prefix # help>;
  def fno_#NAME : Flag<["-"], "fno-"#name>, Flags<[FC1Option] # flags>,
                  Group<f_Group>, HelpText<neg_prefix # help>;
}

// Creates a positive and negative flags where both of them are prefixed with
// "m", have help text specified for positive and negative option, and a Group
// optionally specified by the opt_group argument, otherwise Group<m_Group>.
multiclass SimpleMFlag<string name, string pos_prefix, string neg_prefix,
                       string help, OptionGroup opt_group = m_Group> {
  def m#NAME : Flag<["-"], "m"#name>, Group<opt_group>,
    HelpText<pos_prefix # help>;
  def mno_#NAME : Flag<["-"], "mno-"#name>, Group<opt_group>,
    HelpText<neg_prefix # help>;
}

//===----------------------------------------------------------------------===//
// BoolOption
//===----------------------------------------------------------------------===//

// The default value of a marshalled key path.
class Default<code value> { code Value = value; }

// Convenience variables for boolean defaults.
def DefaultTrue : Default<"true"> {}
def DefaultFalse : Default<"false"> {}

// The value set to the key path when the flag is present on the command line.
class Set<bit value> { bit Value = value; }
def SetTrue : Set<true> {}
def SetFalse : Set<false> {}

// Definition of single command line flag. This is an implementation detail, use
// SetTrueBy or SetFalseBy instead.
class FlagDef<bit polarity, bit value, list<OptionFlag> option_flags,
              string help, list<code> implied_by_expressions = []> {
  // The polarity. Besides spelling, this also decides whether the TableGen
  // record will be prefixed with "no_".
  bit Polarity = polarity;

  // The value assigned to key path when the flag is present on command line.
  bit Value = value;

  // OptionFlags that control visibility of the flag in different tools.
  list<OptionFlag> OptionFlags = option_flags;

  // The help text associated with the flag.
  string Help = help;

  // List of expressions that, when true, imply this flag.
  list<code> ImpliedBy = implied_by_expressions;
}

// Additional information to be appended to both positive and negative flag.
class BothFlags<list<OptionFlag> option_flags, string help = ""> {
  list<OptionFlag> OptionFlags = option_flags;
  string Help = help;
}

// Functor that appends the suffix to the base flag definition.
class ApplySuffix<FlagDef flag, BothFlags suffix> {
  FlagDef Result
    = FlagDef<flag.Polarity, flag.Value,
              flag.OptionFlags # suffix.OptionFlags,
              flag.Help # suffix.Help, flag.ImpliedBy>;
}

// Definition of the command line flag with positive spelling, e.g. "-ffoo".
class PosFlag<Set value, list<OptionFlag> flags = [], string help = "",
              list<code> implied_by_expressions = []>
  : FlagDef<true, value.Value, flags, help, implied_by_expressions> {}

// Definition of the command line flag with negative spelling, e.g. "-fno-foo".
class NegFlag<Set value, list<OptionFlag> flags = [], string help = "",
              list<code> implied_by_expressions = []>
  : FlagDef<false, value.Value, flags, help, implied_by_expressions> {}

// Expanded FlagDef that's convenient for creation of TableGen records.
class FlagDefExpanded<FlagDef flag, string prefix, string name, string spelling>
  : FlagDef<flag.Polarity, flag.Value, flag.OptionFlags, flag.Help,
            flag.ImpliedBy> {
  // Name of the TableGen record.
  string RecordName = prefix # !if(flag.Polarity, "", "no_") # name;

  // Spelling of the flag.
  string Spelling = prefix # !if(flag.Polarity, "", "no-") # spelling;

  // Can the flag be implied by another flag?
  bit CanBeImplied = !not(!empty(flag.ImpliedBy));

  // C++ code that will be assigned to the keypath when the flag is present.
  code ValueAsCode = !if(flag.Value, "true", "false");
}

// TableGen record for a single marshalled flag.
class MarshalledFlagRec<FlagDefExpanded flag, FlagDefExpanded other,
                        FlagDefExpanded implied, KeyPathAndMacro kpm,
                        Default default>
  : Flag<["-"], flag.Spelling>, Flags<flag.OptionFlags>, HelpText<flag.Help>,
    MarshallingInfoBooleanFlag<kpm, default.Value, flag.ValueAsCode,
                               other.ValueAsCode, other.RecordName>,
    ImpliedByAnyOf<implied.ImpliedBy, implied.ValueAsCode> {}

// Generates TableGen records for two command line flags that control the same
// key path via the marshalling infrastructure.
// Names of the records consist of the specified prefix, "no_" for the negative
// flag, and NAME.
// Used for -cc1 frontend options. Driver-only options do not map to
// CompilerInvocation.
multiclass BoolOption<string prefix = "", string spelling_base,
                      KeyPathAndMacro kpm, Default default,
                      FlagDef flag1_base, FlagDef flag2_base,
                      BothFlags suffix = BothFlags<[], "">> {
  defvar flag1 = FlagDefExpanded<ApplySuffix<flag1_base, suffix>.Result, prefix,
                                 NAME, spelling_base>;

  defvar flag2 = FlagDefExpanded<ApplySuffix<flag2_base, suffix>.Result, prefix,
                                 NAME, spelling_base>;

  // The flags must have different polarity, different values, and only
  // one can be implied.
  assert !xor(flag1.Polarity, flag2.Polarity),
         "the flags must have different polarity: flag1: " #
             flag1.Polarity # ", flag2: " # flag2.Polarity;
  assert !ne(flag1.Value, flag2.Value),
         "the flags must have different values: flag1: " #
             flag1.Value # ", flag2: " # flag2.Value;
  assert !not(!and(flag1.CanBeImplied, flag2.CanBeImplied)),
         "only one of the flags can be implied: flag1: " #
             flag1.CanBeImplied # ", flag2: " # flag2.CanBeImplied;

  defvar implied = !if(flag1.CanBeImplied, flag1, flag2);

  def flag1.RecordName : MarshalledFlagRec<flag1, flag2, implied, kpm, default>;
  def flag2.RecordName : MarshalledFlagRec<flag2, flag1, implied, kpm, default>;
}

/// Creates a BoolOption where both of the flags are prefixed with "f", are in
/// the Group<f_Group>.
/// Used for -cc1 frontend options. Driver-only options do not map to
/// CompilerInvocation.
multiclass BoolFOption<string flag_base, KeyPathAndMacro kpm,
                       Default default, FlagDef flag1, FlagDef flag2,
                       BothFlags both = BothFlags<[], "">> {
  defm NAME : BoolOption<"f", flag_base, kpm, default, flag1, flag2, both>,
              Group<f_Group>;
}

// Creates a BoolOption where both of the flags are prefixed with "g" and have
// the Group<g_Group>.
// Used for -cc1 frontend options. Driver-only options do not map to
// CompilerInvocation.
multiclass BoolGOption<string flag_base, KeyPathAndMacro kpm,
                       Default default, FlagDef flag1, FlagDef flag2,
                       BothFlags both = BothFlags<[], "">> {
  defm NAME : BoolOption<"g", flag_base, kpm, default, flag1, flag2, both>,
              Group<g_Group>;
}

// FIXME: Diagnose if target does not support protected visibility.
class MarshallingInfoVisibility<KeyPathAndMacro kpm, code default>
  : MarshallingInfoEnum<kpm, default>,
    Values<"default,hidden,internal,protected">,
    NormalizedValues<["DefaultVisibility", "HiddenVisibility",
                      "HiddenVisibility", "ProtectedVisibility"]> {}

// Key paths that are constant during parsing of options with the same key path prefix.
defvar cplusplus = LangOpts<"CPlusPlus">;
defvar cpp11 = LangOpts<"CPlusPlus11">;
defvar cpp17 = LangOpts<"CPlusPlus17">;
defvar cpp20 = LangOpts<"CPlusPlus20">;
defvar c99 = LangOpts<"C99">;
defvar c2x = LangOpts<"C2x">;
defvar lang_std = LangOpts<"LangStd">;
defvar open_cl = LangOpts<"OpenCL">;
defvar cuda = LangOpts<"CUDA">;
defvar render_script = LangOpts<"RenderScript">;
defvar hip = LangOpts<"HIP">;
defvar gnu_mode = LangOpts<"GNUMode">;
defvar asm_preprocessor = LangOpts<"AsmPreprocessor">;

defvar std = !strconcat("LangStandard::getLangStandardForKind(", lang_std.KeyPath, ")");

/////////
// Options

// The internal option ID must be a valid C++ identifier and results in a
// clang::driver::options::OPT_XX enum constant for XX.
//
// We want to unambiguously be able to refer to options from the driver source
// code, for this reason the option name is mangled into an ID. This mangling
// isn't guaranteed to have an inverse, but for practical purposes it does.
//
// The mangling scheme is to ignore the leading '-', and perform the following
// substitutions:
//   _ => __
//   - => _
//   / => _SLASH
//   # => _HASH
//   ? => _QUESTION
//   , => _COMMA
//   = => _EQ
//   C++ => CXX
//   . => _

// Developer Driver Options

def internal_Group : OptionGroup<"<clang internal options>">, Flags<[HelpHidden]>;
def internal_driver_Group : OptionGroup<"<clang driver internal options>">,
  Group<internal_Group>, HelpText<"DRIVER OPTIONS">;
def internal_debug_Group :
  OptionGroup<"<clang debug/development internal options>">,
  Group<internal_Group>, HelpText<"DEBUG/DEVELOPMENT OPTIONS">;

class InternalDriverOpt : Group<internal_driver_Group>,
  Flags<[NoXarchOption, HelpHidden]>;
def driver_mode : Joined<["--"], "driver-mode=">, Group<internal_driver_Group>,
  Flags<[CoreOption, NoXarchOption, HelpHidden]>,
  HelpText<"Set the driver mode to either 'gcc', 'g++', 'cpp', or 'cl'">;
def rsp_quoting : Joined<["--"], "rsp-quoting=">, Group<internal_driver_Group>,
  Flags<[CoreOption, NoXarchOption, HelpHidden]>,
  HelpText<"Set the rsp quoting to either 'posix', or 'windows'">;
def ccc_gcc_name : Separate<["-"], "ccc-gcc-name">, InternalDriverOpt,
  HelpText<"Name for native GCC compiler">,
  MetaVarName<"<gcc-path>">;

class InternalDebugOpt : Group<internal_debug_Group>,
  Flags<[NoXarchOption, HelpHidden, CoreOption]>;
def ccc_install_dir : Separate<["-"], "ccc-install-dir">, InternalDebugOpt,
  HelpText<"Simulate installation in the given directory">;
def ccc_print_phases : Flag<["-"], "ccc-print-phases">, InternalDebugOpt,
  HelpText<"Dump list of actions to perform">;
def ccc_print_bindings : Flag<["-"], "ccc-print-bindings">, InternalDebugOpt,
  HelpText<"Show bindings of tools to actions">;

def ccc_arcmt_check : Flag<["-"], "ccc-arcmt-check">, InternalDriverOpt,
  HelpText<"Check for ARC migration issues that need manual handling">;
def ccc_arcmt_modify : Flag<["-"], "ccc-arcmt-modify">, InternalDriverOpt,
  HelpText<"Apply modifications to files to conform to ARC">;
def ccc_arcmt_migrate : Separate<["-"], "ccc-arcmt-migrate">, InternalDriverOpt,
  HelpText<"Apply modifications and produces temporary files that conform to ARC">;
def arcmt_migrate_report_output : Separate<["-"], "arcmt-migrate-report-output">,
  HelpText<"Output path for the plist report">,  Flags<[CC1Option]>,
  MarshallingInfoString<FrontendOpts<"ARCMTMigrateReportOut">>;
def arcmt_migrate_emit_arc_errors : Flag<["-"], "arcmt-migrate-emit-errors">,
  HelpText<"Emit ARC errors even if the migrator can fix them">, Flags<[CC1Option]>,
  MarshallingInfoFlag<FrontendOpts<"ARCMTMigrateEmitARCErrors">>;
def gen_reproducer_eq: Joined<["-"], "gen-reproducer=">, Flags<[NoArgumentUnused, CoreOption]>,
  HelpText<"Emit reproducer on (option: off, crash (default), error, always)">;
def gen_reproducer: Flag<["-"], "gen-reproducer">, InternalDebugOpt,
  Alias<gen_reproducer_eq>, AliasArgs<["always"]>,
  HelpText<"Auto-generates preprocessed source files and a reproduction script">;
def gen_cdb_fragment_path: Separate<["-"], "gen-cdb-fragment-path">, InternalDebugOpt,
  HelpText<"Emit a compilation database fragment to the specified directory">;

def round_trip_args : Flag<["-"], "round-trip-args">, Flags<[CC1Option, NoDriverOption]>,
  HelpText<"Enable command line arguments round-trip.">;
def no_round_trip_args : Flag<["-"], "no-round-trip-args">, Flags<[CC1Option, NoDriverOption]>,
  HelpText<"Disable command line arguments round-trip.">;

def _migrate : Flag<["--"], "migrate">, Flags<[NoXarchOption]>,
  HelpText<"Run the migrator">;
def ccc_objcmt_migrate : Separate<["-"], "ccc-objcmt-migrate">,
  InternalDriverOpt,
  HelpText<"Apply modifications and produces temporary files to migrate to "
   "modern ObjC syntax">;

def objcmt_migrate_literals : Flag<["-"], "objcmt-migrate-literals">, Flags<[CC1Option]>,
  HelpText<"Enable migration to modern ObjC literals">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_Literals">;
def objcmt_migrate_subscripting : Flag<["-"], "objcmt-migrate-subscripting">, Flags<[CC1Option]>,
  HelpText<"Enable migration to modern ObjC subscripting">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_Subscripting">;
def objcmt_migrate_property : Flag<["-"], "objcmt-migrate-property">, Flags<[CC1Option]>,
  HelpText<"Enable migration to modern ObjC property">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_Property">;
def objcmt_migrate_all : Flag<["-"], "objcmt-migrate-all">, Flags<[CC1Option]>,
  HelpText<"Enable migration to modern ObjC">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_MigrateDecls">;
def objcmt_migrate_readonly_property : Flag<["-"], "objcmt-migrate-readonly-property">, Flags<[CC1Option]>,
  HelpText<"Enable migration to modern ObjC readonly property">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_ReadonlyProperty">;
def objcmt_migrate_readwrite_property : Flag<["-"], "objcmt-migrate-readwrite-property">, Flags<[CC1Option]>,
  HelpText<"Enable migration to modern ObjC readwrite property">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_ReadwriteProperty">;
def objcmt_migrate_property_dot_syntax : Flag<["-"], "objcmt-migrate-property-dot-syntax">, Flags<[CC1Option]>,
  HelpText<"Enable migration of setter/getter messages to property-dot syntax">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_PropertyDotSyntax">;
def objcmt_migrate_annotation : Flag<["-"], "objcmt-migrate-annotation">, Flags<[CC1Option]>,
  HelpText<"Enable migration to property and method annotations">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_Annotation">;
def objcmt_migrate_instancetype : Flag<["-"], "objcmt-migrate-instancetype">, Flags<[CC1Option]>,
  HelpText<"Enable migration to infer instancetype for method result type">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_Instancetype">;
def objcmt_migrate_nsmacros : Flag<["-"], "objcmt-migrate-ns-macros">, Flags<[CC1Option]>,
  HelpText<"Enable migration to NS_ENUM/NS_OPTIONS macros">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_NsMacros">;
def objcmt_migrate_protocol_conformance : Flag<["-"], "objcmt-migrate-protocol-conformance">, Flags<[CC1Option]>,
  HelpText<"Enable migration to add protocol conformance on classes">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_ProtocolConformance">;
def objcmt_atomic_property : Flag<["-"], "objcmt-atomic-property">, Flags<[CC1Option]>,
  HelpText<"Make migration to 'atomic' properties">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_AtomicProperty">;
def objcmt_returns_innerpointer_property : Flag<["-"], "objcmt-returns-innerpointer-property">, Flags<[CC1Option]>,
  HelpText<"Enable migration to annotate property with NS_RETURNS_INNER_POINTER">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_ReturnsInnerPointerProperty">;
def objcmt_ns_nonatomic_iosonly: Flag<["-"], "objcmt-ns-nonatomic-iosonly">, Flags<[CC1Option]>,
  HelpText<"Enable migration to use NS_NONATOMIC_IOSONLY macro for setting property's 'atomic' attribute">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_NsAtomicIOSOnlyProperty">;
def objcmt_migrate_designated_init : Flag<["-"], "objcmt-migrate-designated-init">, Flags<[CC1Option]>,
  HelpText<"Enable migration to infer NS_DESIGNATED_INITIALIZER for initializer methods">,
  MarshallingInfoBitfieldFlag<FrontendOpts<"ObjCMTAction">, "FrontendOptions::ObjCMT_DesignatedInitializer">;

def objcmt_allowlist_dir_path: Joined<["-"], "objcmt-allowlist-dir-path=">, Flags<[CC1Option]>,
  HelpText<"Only modify files with a filename contained in the provided directory path">,
  MarshallingInfoString<FrontendOpts<"ObjCMTAllowListPath">>;
def : Joined<["-"], "objcmt-whitelist-dir-path=">, Flags<[CC1Option]>,
  HelpText<"Alias for -objcmt-allowlist-dir-path">,
  Alias<objcmt_allowlist_dir_path>;
// The misspelt "white-list" [sic] alias is due for removal.
def : Joined<["-"], "objcmt-white-list-dir-path=">, Flags<[CC1Option]>,
  Alias<objcmt_allowlist_dir_path>;

// Make sure all other -ccc- options are rejected.
def ccc_ : Joined<["-"], "ccc-">, Group<internal_Group>, Flags<[Unsupported]>;

// Standard Options

def _HASH_HASH_HASH : Flag<["-"], "###">, Flags<[NoXarchOption, CoreOption, FlangOption]>,
    HelpText<"Print (but do not run) the commands to run for this compilation">;
def _DASH_DASH : Option<["--"], "", KIND_REMAINING_ARGS>,
    Flags<[NoXarchOption, CoreOption]>;
def A : JoinedOrSeparate<["-"], "A">, Flags<[RenderJoined]>, Group<gfortran_Group>;
def B : JoinedOrSeparate<["-"], "B">, MetaVarName<"<prefix>">,
    HelpText<"Search $prefix$file for executables, libraries, and data files. "
    "If $prefix is a directory, search $prefix/$file">;
def gcc_toolchain : Joined<["--"], "gcc-toolchain=">, Flags<[NoXarchOption]>,
  HelpText<"Search for GCC installation in the specified directory on targets which commonly use GCC. "
  "The directory usually contains 'lib{,32,64}/gcc{,-cross}/$triple' and 'include'. If specified, "
  "sysroot is skipped for GCC detection. Note: executables (e.g. ld) used by the compiler are not "
  "overridden by the selected GCC installation">;
def CC : Flag<["-"], "CC">, Flags<[CC1Option]>, Group<Preprocessor_Group>,
    HelpText<"Include comments from within macros in preprocessed output">,
    MarshallingInfoFlag<PreprocessorOutputOpts<"ShowMacroComments">>;
def C : Flag<["-"], "C">, Flags<[CC1Option]>, Group<Preprocessor_Group>,
    HelpText<"Include comments in preprocessed output">,
    MarshallingInfoFlag<PreprocessorOutputOpts<"ShowComments">>;
def D : JoinedOrSeparate<["-"], "D">, Group<Preprocessor_Group>,
    Flags<[CC1Option, FlangOption, FC1Option]>, MetaVarName<"<macro>=<value>">,
    HelpText<"Define <macro> to <value> (or 1 if <value> omitted)">;
def E : Flag<["-"], "E">, Flags<[NoXarchOption,CC1Option, FlangOption, FC1Option]>, Group<Action_Group>,
    HelpText<"Only run the preprocessor">;
def F : JoinedOrSeparate<["-"], "F">, Flags<[RenderJoined,CC1Option]>,
    HelpText<"Add directory to framework include search path">;
def G : JoinedOrSeparate<["-"], "G">, Flags<[NoXarchOption]>, Group<m_Group>,
    MetaVarName<"<size>">, HelpText<"Put objects of at most <size> bytes "
    "into small data section (MIPS / Hexagon)">;
def G_EQ : Joined<["-"], "G=">, Flags<[NoXarchOption]>, Group<m_Group>, Alias<G>;
def H : Flag<["-"], "H">, Flags<[CC1Option]>, Group<Preprocessor_Group>,
    HelpText<"Show header includes and nesting depth">,
    MarshallingInfoFlag<DependencyOutputOpts<"ShowHeaderIncludes">>;
def fshow_skipped_includes : Flag<["-"], "fshow-skipped-includes">,
  Flags<[CC1Option]>, HelpText<"Show skipped includes in -H output.">,
  DocBrief<[{#include files may be "skipped" due to include guard optimization
             or #pragma once. This flag makes -H show also such includes.}]>,
  MarshallingInfoFlag<DependencyOutputOpts<"ShowSkippedHeaderIncludes">>;

def I_ : Flag<["-"], "I-">, Group<I_Group>,
    HelpText<"Restrict all prior -I flags to double-quoted inclusion and "
             "remove current directory from include path">;
def I : JoinedOrSeparate<["-"], "I">, Group<I_Group>,
    Flags<[CC1Option,CC1AsOption,FlangOption,FC1Option]>, MetaVarName<"<dir>">,
    HelpText<"Add directory to the end of the list of include search paths">,
    DocBrief<[{Add directory to include search path. For C++ inputs, if
there are multiple -I options, these directories are searched
in the order they are given before the standard system directories
are searched. If the same directory is in the SYSTEM include search
paths, for example if also specified with -isystem, the -I option
will be ignored}]>;
def L : JoinedOrSeparate<["-"], "L">, Flags<[RenderJoined]>, Group<Link_Group>,
    MetaVarName<"<dir>">, HelpText<"Add directory to library search path">;
def MD : Flag<["-"], "MD">, Group<M_Group>,
    HelpText<"Write a depfile containing user and system headers">;
def MMD : Flag<["-"], "MMD">, Group<M_Group>,
    HelpText<"Write a depfile containing user headers">;
def M : Flag<["-"], "M">, Group<M_Group>,
    HelpText<"Like -MD, but also implies -E and writes to stdout by default">;
def MM : Flag<["-"], "MM">, Group<M_Group>,
    HelpText<"Like -MMD, but also implies -E and writes to stdout by default">;
def MF : JoinedOrSeparate<["-"], "MF">, Group<M_Group>,
    HelpText<"Write depfile output from -MMD, -MD, -MM, or -M to <file>">,
    MetaVarName<"<file>">;
def MG : Flag<["-"], "MG">, Group<M_Group>, Flags<[CC1Option]>,
    HelpText<"Add missing headers to depfile">,
    MarshallingInfoFlag<DependencyOutputOpts<"AddMissingHeaderDeps">>;
def MJ : JoinedOrSeparate<["-"], "MJ">, Group<M_Group>,
    HelpText<"Write a compilation database entry per input">;
def MP : Flag<["-"], "MP">, Group<M_Group>, Flags<[CC1Option]>,
    HelpText<"Create phony target for each dependency (other than main file)">,
    MarshallingInfoFlag<DependencyOutputOpts<"UsePhonyTargets">>;
def MQ : JoinedOrSeparate<["-"], "MQ">, Group<M_Group>, Flags<[CC1Option]>,
    HelpText<"Specify name of main file output to quote in depfile">;
def MT : JoinedOrSeparate<["-"], "MT">, Group<M_Group>, Flags<[CC1Option]>,
    HelpText<"Specify name of main file output in depfile">,
    MarshallingInfoStringVector<DependencyOutputOpts<"Targets">>;
def MV : Flag<["-"], "MV">, Group<M_Group>, Flags<[CC1Option]>,
    HelpText<"Use NMake/Jom format for the depfile">,
    MarshallingInfoFlag<DependencyOutputOpts<"OutputFormat">, "DependencyOutputFormat::Make">,
    Normalizer<"makeFlagToValueNormalizer(DependencyOutputFormat::NMake)">;
def Mach : Flag<["-"], "Mach">, Group<Link_Group>;
def O0 : Flag<["-"], "O0">, Group<O_Group>, Flags<[CC1Option, FC1Option, HelpHidden]>;
def O4 : Flag<["-"], "O4">, Group<O_Group>, Flags<[CC1Option, FC1Option, HelpHidden]>;
def ObjCXX : Flag<["-"], "ObjC++">, Flags<[NoXarchOption]>,
  HelpText<"Treat source input files as Objective-C++ inputs">;
def ObjC : Flag<["-"], "ObjC">, Flags<[NoXarchOption]>,
  HelpText<"Treat source input files as Objective-C inputs">;
def O : Joined<["-"], "O">, Group<O_Group>, Flags<[CC1Option,FC1Option]>;
def O_flag : Flag<["-"], "O">, Flags<[CC1Option,FC1Option]>, Alias<O>, AliasArgs<["1"]>;
def Ofast : Joined<["-"], "Ofast">, Group<O_Group>, Flags<[CC1Option]>;
def P : Flag<["-"], "P">, Flags<[CC1Option,FlangOption,FC1Option]>, Group<Preprocessor_Group>,
  HelpText<"Disable linemarker output in -E mode">,
  MarshallingInfoNegativeFlag<PreprocessorOutputOpts<"ShowLineMarkers">>;
def Qy : Flag<["-"], "Qy">, Flags<[CC1Option]>,
  HelpText<"Emit metadata containing compiler name and version">;
def Qn : Flag<["-"], "Qn">, Flags<[CC1Option]>,
  HelpText<"Do not emit metadata containing compiler name and version">;
def : Flag<["-"], "fident">, Group<f_Group>, Alias<Qy>,
  Flags<[CoreOption, CC1Option]>;
def : Flag<["-"], "fno-ident">, Group<f_Group>, Alias<Qn>,
  Flags<[CoreOption, CC1Option]>;
def Qunused_arguments : Flag<["-"], "Qunused-arguments">, Flags<[NoXarchOption, CoreOption]>,
  HelpText<"Don't emit warning for unused driver arguments">;
def Q : Flag<["-"], "Q">, IgnoredGCCCompat;
def Rpass_EQ : Joined<["-"], "Rpass=">, Group<R_value_Group>, Flags<[CC1Option]>,
  HelpText<"Report transformations performed by optimization passes whose "
           "name matches the given POSIX regular expression">;
def Rpass_missed_EQ : Joined<["-"], "Rpass-missed=">, Group<R_value_Group>,
  Flags<[CC1Option]>,
  HelpText<"Report missed transformations by optimization passes whose "
           "name matches the given POSIX regular expression">;
def Rpass_analysis_EQ : Joined<["-"], "Rpass-analysis=">, Group<R_value_Group>,
  Flags<[CC1Option]>,
  HelpText<"Report transformation analysis from optimization passes whose "
           "name matches the given POSIX regular expression">;
def R_Joined : Joined<["-"], "R">, Group<R_Group>, Flags<[CC1Option, CoreOption]>,
  MetaVarName<"<remark>">, HelpText<"Enable the specified remark">;
def S : Flag<["-"], "S">, Flags<[NoXarchOption,CC1Option,FlangOption,FC1Option]>, Group<Action_Group>,
  HelpText<"Only run preprocess and compilation steps">;
def Tbss : JoinedOrSeparate<["-"], "Tbss">, Group<T_Group>,
  MetaVarName<"<addr>">, HelpText<"Set starting address of BSS to <addr>">;
def Tdata : JoinedOrSeparate<["-"], "Tdata">, Group<T_Group>,
  MetaVarName<"<addr>">, HelpText<"Set starting address of DATA to <addr>">;
def Ttext : JoinedOrSeparate<["-"], "Ttext">, Group<T_Group>,
  MetaVarName<"<addr>">, HelpText<"Set starting address of TEXT to <addr>">;
def T : JoinedOrSeparate<["-"], "T">, Group<T_Group>,
  MetaVarName<"<script>">, HelpText<"Specify <script> as linker script">;
def U : JoinedOrSeparate<["-"], "U">, Group<Preprocessor_Group>,
  Flags<[CC1Option, FlangOption, FC1Option]>, MetaVarName<"<macro>">, HelpText<"Undefine macro <macro>">;
def V : JoinedOrSeparate<["-"], "V">, Flags<[NoXarchOption, Unsupported]>;
def Wa_COMMA : CommaJoined<["-"], "Wa,">,
  HelpText<"Pass the comma separated arguments in <arg> to the assembler">,
  MetaVarName<"<arg>">;
def Wall : Flag<["-"], "Wall">, Group<W_Group>, Flags<[CC1Option, HelpHidden]>;
def WCL4 : Flag<["-"], "WCL4">, Group<W_Group>, Flags<[CC1Option, HelpHidden]>;
def Wsystem_headers : Flag<["-"], "Wsystem-headers">, Group<W_Group>, Flags<[CC1Option, HelpHidden]>;
def Wno_system_headers : Flag<["-"], "Wno-system-headers">, Group<W_Group>, Flags<[CC1Option, HelpHidden]>;
def Wdeprecated : Flag<["-"], "Wdeprecated">, Group<W_Group>, Flags<[CC1Option]>,
  HelpText<"Enable warnings for deprecated constructs and define __DEPRECATED">;
def Wno_deprecated : Flag<["-"], "Wno-deprecated">, Group<W_Group>, Flags<[CC1Option]>;
def Wl_COMMA : CommaJoined<["-"], "Wl,">, Flags<[LinkerInput, RenderAsInput]>,
  HelpText<"Pass the comma separated arguments in <arg> to the linker">,
  MetaVarName<"<arg>">, Group<Link_Group>;
// FIXME: This is broken; these should not be Joined arguments.
def Wno_nonportable_cfstrings : Joined<["-"], "Wno-nonportable-cfstrings">, Group<W_Group>,
  Flags<[CC1Option]>;
def Wnonportable_cfstrings : Joined<["-"], "Wnonportable-cfstrings">, Group<W_Group>,
  Flags<[CC1Option]>;
def Wp_COMMA : CommaJoined<["-"], "Wp,">,
  HelpText<"Pass the comma separated arguments in <arg> to the preprocessor">,
  MetaVarName<"<arg>">, Group<Preprocessor_Group>;
def Wundef_prefix_EQ : CommaJoined<["-"], "Wundef-prefix=">, Group<W_value_Group>,
  Flags<[CC1Option, CoreOption, HelpHidden]>, MetaVarName<"<arg>">,
  HelpText<"Enable warnings for undefined macros with a prefix in the comma separated list <arg>">,
  MarshallingInfoStringVector<DiagnosticOpts<"UndefPrefixes">>;
def Wwrite_strings : Flag<["-"], "Wwrite-strings">, Group<W_Group>, Flags<[CC1Option, HelpHidden]>;
def Wno_write_strings : Flag<["-"], "Wno-write-strings">, Group<W_Group>, Flags<[CC1Option, HelpHidden]>;
def W_Joined : Joined<["-"], "W">, Group<W_Group>, Flags<[CC1Option, CoreOption, FC1Option, FlangOption]>,
  MetaVarName<"<warning>">, HelpText<"Enable the specified warning">;
def Xanalyzer : Separate<["-"], "Xanalyzer">,
  HelpText<"Pass <arg> to the static analyzer">, MetaVarName<"<arg>">,
  Group<StaticAnalyzer_Group>;
def Xarch__ : JoinedAndSeparate<["-"], "Xarch_">, Flags<[NoXarchOption]>;
def Xarch_host : Separate<["-"], "Xarch_host">, Flags<[NoXarchOption]>,
  HelpText<"Pass <arg> to the CUDA/HIP host compilation">, MetaVarName<"<arg>">;
def Xarch_device : Separate<["-"], "Xarch_device">, Flags<[NoXarchOption]>,
  HelpText<"Pass <arg> to the CUDA/HIP device compilation">, MetaVarName<"<arg>">;
def Xassembler : Separate<["-"], "Xassembler">,
  HelpText<"Pass <arg> to the assembler">, MetaVarName<"<arg>">,
  Group<CompileOnly_Group>;
def Xclang : Separate<["-"], "Xclang">,
  HelpText<"Pass <arg> to the clang compiler">, MetaVarName<"<arg>">,
  Flags<[NoXarchOption, CoreOption]>, Group<CompileOnly_Group>;
def Xcuda_fatbinary : Separate<["-"], "Xcuda-fatbinary">,
  HelpText<"Pass <arg> to fatbinary invocation">, MetaVarName<"<arg>">;
def Xcuda_ptxas : Separate<["-"], "Xcuda-ptxas">,
  HelpText<"Pass <arg> to the ptxas assembler">, MetaVarName<"<arg>">;
def Xopenmp_target : Separate<["-"], "Xopenmp-target">, Group<CompileOnly_Group>,
  HelpText<"Pass <arg> to the target offloading toolchain.">, MetaVarName<"<arg>">;
def Xopenmp_target_EQ : JoinedAndSeparate<["-"], "Xopenmp-target=">, Group<CompileOnly_Group>,
  HelpText<"Pass <arg> to the target offloading toolchain identified by <triple>.">,
  MetaVarName<"<triple> <arg>">;
def z : Separate<["-"], "z">, Flags<[LinkerInput, RenderAsInput]>,
  HelpText<"Pass -z <arg> to the linker">, MetaVarName<"<arg>">,
  Group<Link_Group>;
def offload_link : Flag<["--"], "offload-link">, Group<Link_Group>,
  HelpText<"Use the new offloading linker to perform the link job.">;
def Xlinker : Separate<["-"], "Xlinker">, Flags<[LinkerInput, RenderAsInput]>,
  HelpText<"Pass <arg> to the linker">, MetaVarName<"<arg>">,
  Group<Link_Group>;
def Xoffload_linker : JoinedAndSeparate<["-"], "Xoffload-linker">,
  HelpText<"Pass <arg> to the offload linkers or the ones idenfied by -<triple>">,
  MetaVarName<"<triple> <arg>">, Group<Link_Group>;
def Xpreprocessor : Separate<["-"], "Xpreprocessor">, Group<Preprocessor_Group>,
  HelpText<"Pass <arg> to the preprocessor">, MetaVarName<"<arg>">;
def X_Flag : Flag<["-"], "X">, Group<Link_Group>;
def X_Joined : Joined<["-"], "X">, IgnoredGCCCompat;
def Z_Flag : Flag<["-"], "Z">, Group<Link_Group>;
// FIXME: All we do with this is reject it. Remove.
def Z_Joined : Joined<["-"], "Z">;
def all__load : Flag<["-"], "all_load">;
def allowable__client : Separate<["-"], "allowable_client">;
def ansi : Flag<["-", "--"], "ansi">, Group<CompileOnly_Group>;
def arch__errors__fatal : Flag<["-"], "arch_errors_fatal">;
def arch : Separate<["-"], "arch">, Flags<[NoXarchOption]>;
def arch__only : Separate<["-"], "arch_only">;
def a : Joined<["-"], "a">;
def autocomplete : Joined<["--"], "autocomplete=">;
def bind__at__load : Flag<["-"], "bind_at_load">;
def bundle__loader : Separate<["-"], "bundle_loader">;
def bundle : Flag<["-"], "bundle">;
def b : JoinedOrSeparate<["-"], "b">, Flags<[LinkerInput, RenderAsInput]>,
  HelpText<"Pass -b <arg> to the linker on AIX (only).">, MetaVarName<"<arg>">,
  Group<Link_Group>;
// OpenCL-only Options
def cl_opt_disable : Flag<["-"], "cl-opt-disable">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. This option disables all optimizations. By default optimizations are enabled.">;
def cl_strict_aliasing : Flag<["-"], "cl-strict-aliasing">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. This option is added for compatibility with OpenCL 1.0.">;
def cl_single_precision_constant : Flag<["-"], "cl-single-precision-constant">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. Treat double precision floating-point constant as single precision constant.">,
  MarshallingInfoFlag<LangOpts<"SinglePrecisionConstants">>;
def cl_finite_math_only : Flag<["-"], "cl-finite-math-only">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. Allow floating-point optimizations that assume arguments and results are not NaNs or +-Inf.">,
  MarshallingInfoFlag<LangOpts<"CLFiniteMathOnly">>;
def cl_kernel_arg_info : Flag<["-"], "cl-kernel-arg-info">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. Generate kernel argument metadata.">,
  MarshallingInfoFlag<CodeGenOpts<"EmitOpenCLArgMetadata">>;
def cl_unsafe_math_optimizations : Flag<["-"], "cl-unsafe-math-optimizations">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. Allow unsafe floating-point optimizations.  Also implies -cl-no-signed-zeros and -cl-mad-enable.">,
  MarshallingInfoFlag<LangOpts<"CLUnsafeMath">>;
def cl_fast_relaxed_math : Flag<["-"], "cl-fast-relaxed-math">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. Sets -cl-finite-math-only and -cl-unsafe-math-optimizations, and defines __FAST_RELAXED_MATH__.">,
  MarshallingInfoFlag<LangOpts<"FastRelaxedMath">>;
def cl_mad_enable : Flag<["-"], "cl-mad-enable">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. Allow use of less precise MAD computations in the generated binary.">,
  MarshallingInfoFlag<CodeGenOpts<"LessPreciseFPMAD">>,
  ImpliedByAnyOf<[cl_unsafe_math_optimizations.KeyPath, cl_fast_relaxed_math.KeyPath]>;
def cl_no_signed_zeros : Flag<["-"], "cl-no-signed-zeros">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. Allow use of less precise no signed zeros computations in the generated binary.">,
  MarshallingInfoFlag<LangOpts<"CLNoSignedZero">>;
def cl_std_EQ : Joined<["-"], "cl-std=">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL language standard to compile for.">,
  Values<"cl,CL,cl1.0,CL1.0,cl1.1,CL1.1,cl1.2,CL1.2,cl2.0,CL2.0,cl3.0,CL3.0,clc++,CLC++,clc++1.0,CLC++1.0,clc++2021,CLC++2021">;
def cl_denorms_are_zero : Flag<["-"], "cl-denorms-are-zero">, Group<opencl_Group>,
  HelpText<"OpenCL only. Allow denormals to be flushed to zero.">;
def cl_fp32_correctly_rounded_divide_sqrt : Flag<["-"], "cl-fp32-correctly-rounded-divide-sqrt">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. Specify that single precision floating-point divide and sqrt used in the program source are correctly rounded.">,
  MarshallingInfoFlag<CodeGenOpts<"OpenCLCorrectlyRoundedDivSqrt">>;
def cl_uniform_work_group_size : Flag<["-"], "cl-uniform-work-group-size">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. Defines that the global work-size be a multiple of the work-group size specified to clEnqueueNDRangeKernel">,
  MarshallingInfoFlag<CodeGenOpts<"UniformWGSize">>;
def cl_no_stdinc : Flag<["-"], "cl-no-stdinc">, Group<opencl_Group>,
  HelpText<"OpenCL only. Disables all standard includes containing non-native compiler types and functions.">;
def cl_ext_EQ : CommaJoined<["-"], "cl-ext=">, Group<opencl_Group>, Flags<[CC1Option]>,
  HelpText<"OpenCL only. Enable or disable OpenCL extensions/optional features. The argument is a comma-separated "
           "sequence of one or more extension names, each prefixed by '+' or '-'.">,
  MarshallingInfoStringVector<TargetOpts<"OpenCLExtensionsAsWritten">>;

def client__name : JoinedOrSeparate<["-"], "client_name">;
def combine : Flag<["-", "--"], "combine">, Flags<[NoXarchOption, Unsupported]>;
def compatibility__version : JoinedOrSeparate<["-"], "compatibility_version">;
def config : Separate<["--"], "config">, Flags<[NoXarchOption]>,
  HelpText<"Specifies configuration file">;
def config_system_dir_EQ : Joined<["--"], "config-system-dir=">, Flags<[NoXarchOption, HelpHidden]>,
  HelpText<"System directory for configuration files">;
def config_user_dir_EQ : Joined<["--"], "config-user-dir=">, Flags<[NoXarchOption, HelpHidden]>,
  HelpText<"User directory for configuration files">;
def coverage : Flag<["-", "--"], "coverage">, Group<Link_Group>, Flags<[CoreOption]>;
def cpp_precomp : Flag<["-"], "cpp-precomp">, Group<clang_ignored_f_Group>;
def current__version : JoinedOrSeparate<["-"], "current_version">;
def cxx_isystem : JoinedOrSeparate<["-"], "cxx-isystem">, Group<clang_i_Group>,
  HelpText<"Add directory to the C++ SYSTEM include search path">, Flags<[CC1Option]>,
  MetaVarName<"<directory>">;
def c : Flag<["-"], "c">, Flags<[NoXarchOption, FlangOption]>, Group<Action_Group>,
  HelpText<"Only run preprocess, compile, and assemble steps">;
def fconvergent_functions : Flag<["-"], "fconvergent-functions">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Assume functions may be convergent">;

def gpu_use_aux_triple_only : Flag<["--"], "gpu-use-aux-triple-only">,
  InternalDriverOpt, HelpText<"Prepare '-aux-triple' only without populating "
                              "'-aux-target-cpu' and '-aux-target-feature'.">;
def cuda_include_ptx_EQ : Joined<["--"], "cuda-include-ptx=">, Flags<[NoXarchOption]>,
  HelpText<"Include PTX for the following GPU architecture (e.g. sm_35) or 'all'. May be specified more than once.">;
def no_cuda_include_ptx_EQ : Joined<["--"], "no-cuda-include-ptx=">, Flags<[NoXarchOption]>,
  HelpText<"Do not include PTX for the following GPU architecture (e.g. sm_35) or 'all'. May be specified more than once.">;
def offload_arch_EQ : Joined<["--"], "offload-arch=">, Flags<[NoXarchOption]>,
  HelpText<"CUDA offloading device architecture (e.g. sm_35), or HIP offloading target ID in the form of a "
           "device architecture followed by target ID features delimited by a colon. Each target ID feature "
           "is a pre-defined string followed by a plus or minus sign (e.g. gfx908:xnack+:sramecc-).  May be "
           "specified more than once.">;
def cuda_gpu_arch_EQ : Joined<["--"], "cuda-gpu-arch=">, Flags<[NoXarchOption]>,
  Alias<offload_arch_EQ>;
def cuda_feature_EQ : Joined<["--"], "cuda-feature=">, HelpText<"Manually specify the CUDA feature to use">;
def hip_link : Flag<["--"], "hip-link">,
  HelpText<"Link clang-offload-bundler bundles for HIP">;
def no_hip_rt: Flag<["-"], "no-hip-rt">,
  HelpText<"Do not link against HIP runtime libraries">;
def no_offload_arch_EQ : Joined<["--"], "no-offload-arch=">, Flags<[NoXarchOption]>,
  HelpText<"Remove CUDA/HIP offloading device architecture (e.g. sm_35, gfx906) from the list of devices to compile for. "
           "'all' resets the list to its default value.">;
def emit_static_lib : Flag<["--"], "emit-static-lib">,
  HelpText<"Enable linker job to emit a static library.">;
def no_cuda_gpu_arch_EQ : Joined<["--"], "no-cuda-gpu-arch=">, Flags<[NoXarchOption]>,
  Alias<no_offload_arch_EQ>;
def cuda_noopt_device_debug : Flag<["--"], "cuda-noopt-device-debug">,
  HelpText<"Enable device-side debug info generation. Disables ptxas optimizations.">;
def no_cuda_version_check : Flag<["--"], "no-cuda-version-check">,
  HelpText<"Don't error out if the detected version of the CUDA install is "
           "too low for the requested CUDA gpu architecture.">;
def no_cuda_noopt_device_debug : Flag<["--"], "no-cuda-noopt-device-debug">;
def cuda_path_EQ : Joined<["--"], "cuda-path=">, Group<i_Group>,
  HelpText<"CUDA installation path">;
def cuda_path_ignore_env : Flag<["--"], "cuda-path-ignore-env">, Group<i_Group>,
  HelpText<"Ignore environment variables to detect CUDA installation">;
def ptxas_path_EQ : Joined<["--"], "ptxas-path=">, Group<i_Group>,
  HelpText<"Path to ptxas (used for compiling CUDA code)">;
def fgpu_flush_denormals_to_zero : Flag<["-"], "fgpu-flush-denormals-to-zero">,
  HelpText<"Flush denormal floating point values to zero in CUDA/HIP device mode.">;
def fno_gpu_flush_denormals_to_zero : Flag<["-"], "fno-gpu-flush-denormals-to-zero">;
def fcuda_flush_denormals_to_zero : Flag<["-"], "fcuda-flush-denormals-to-zero">,
  Alias<fgpu_flush_denormals_to_zero>;
def fno_cuda_flush_denormals_to_zero : Flag<["-"], "fno-cuda-flush-denormals-to-zero">,
  Alias<fno_gpu_flush_denormals_to_zero>;
defm gpu_rdc : BoolFOption<"gpu-rdc",
  LangOpts<"GPURelocatableDeviceCode">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Generate relocatable device code, also known as separate compilation mode">,
  NegFlag<SetFalse>>;
def : Flag<["-"], "fcuda-rdc">, Alias<fgpu_rdc>;
def : Flag<["-"], "fno-cuda-rdc">, Alias<fno_gpu_rdc>;
defm cuda_short_ptr : BoolFOption<"cuda-short-ptr",
  TargetOpts<"NVPTXUseShortPointers">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Use 32-bit pointers for accessing const/local/shared address spaces">,
  NegFlag<SetFalse>>;
def fgpu_default_stream_EQ : Joined<["-"], "fgpu-default-stream=">,
  HelpText<"Specify default stream. The default value is 'legacy'. (HIP only)">,
  Flags<[CC1Option]>,
  Values<"legacy,per-thread">,
  NormalizedValuesScope<"LangOptions::GPUDefaultStreamKind">,
  NormalizedValues<["Legacy", "PerThread"]>,
  MarshallingInfoEnum<LangOpts<"GPUDefaultStream">, "Legacy">;
def rocm_path_EQ : Joined<["--"], "rocm-path=">, Group<i_Group>,
  HelpText<"ROCm installation path, used for finding and automatically linking required bitcode libraries.">;
def hip_path_EQ : Joined<["--"], "hip-path=">, Group<i_Group>,
  HelpText<"HIP runtime installation path, used for finding HIP version and adding HIP include path.">;
def amdgpu_arch_tool_EQ : Joined<["--"], "amdgpu-arch-tool=">, Group<i_Group>,
  HelpText<"Tool used for detecting AMD GPU arch in the system.">;
def rocm_device_lib_path_EQ : Joined<["--"], "rocm-device-lib-path=">, Group<Link_Group>,
  HelpText<"ROCm device library path. Alternative to rocm-path.">;
def : Joined<["--"], "hip-device-lib-path=">, Alias<rocm_device_lib_path_EQ>;
def hip_device_lib_EQ : Joined<["--"], "hip-device-lib=">, Group<Link_Group>,
  HelpText<"HIP device library">;
def hip_version_EQ : Joined<["--"], "hip-version=">,
  HelpText<"HIP version in the format of major.minor.patch">;
def fhip_dump_offload_linker_script : Flag<["-"], "fhip-dump-offload-linker-script">,
  Group<f_Group>, Flags<[NoArgumentUnused, HelpHidden]>;
defm hip_new_launch_api : BoolFOption<"hip-new-launch-api",
  LangOpts<"HIPUseNewLaunchAPI">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Use">, NegFlag<SetFalse, [], "Don't use">,
  BothFlags<[], " new kernel launching API for HIP">>;
defm hip_fp32_correctly_rounded_divide_sqrt : BoolFOption<"hip-fp32-correctly-rounded-divide-sqrt",
  CodeGenOpts<"HIPCorrectlyRoundedDivSqrt">, DefaultTrue,
  PosFlag<SetTrue, [], "Specify">,
  NegFlag<SetFalse, [CC1Option], "Don't specify">,
  BothFlags<[], " that single precision floating-point divide and sqrt used in "
  "the program source are correctly rounded (HIP device compilation only)">>,
  ShouldParseIf<hip.KeyPath>;
defm hip_kernel_arg_name : BoolFOption<"hip-kernel-arg-name",
  CodeGenOpts<"HIPSaveKernelArgName">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Specify">,
  NegFlag<SetFalse, [], "Don't specify">,
  BothFlags<[], " that kernel argument names are preserved (HIP only)">>,
  ShouldParseIf<hip.KeyPath>;
def hipspv_pass_plugin_EQ : Joined<["--"], "hipspv-pass-plugin=">,
  Group<Link_Group>, MetaVarName<"<dsopath>">,
  HelpText<"path to a pass plugin for HIP to SPIR-V passes.">;
defm gpu_allow_device_init : BoolFOption<"gpu-allow-device-init",
  LangOpts<"GPUAllowDeviceInit">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Allow">, NegFlag<SetFalse, [], "Don't allow">,
  BothFlags<[], " device side init function in HIP (experimental)">>,
  ShouldParseIf<hip.KeyPath>;
defm gpu_defer_diag : BoolFOption<"gpu-defer-diag",
  LangOpts<"GPUDeferDiag">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Defer">, NegFlag<SetFalse, [], "Don't defer">,
  BothFlags<[], " host/device related diagnostic messages for CUDA/HIP">>;
defm gpu_exclude_wrong_side_overloads : BoolFOption<"gpu-exclude-wrong-side-overloads",
  LangOpts<"GPUExcludeWrongSideOverloads">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Always exclude wrong side overloads">,
  NegFlag<SetFalse, [], "Exclude wrong side overloads only if there are same side overloads">,
  BothFlags<[HelpHidden], " in overloading resolution for CUDA/HIP">>;
def gpu_max_threads_per_block_EQ : Joined<["--"], "gpu-max-threads-per-block=">,
  Flags<[CC1Option]>,
  HelpText<"Default max threads per block for kernel launch bounds for HIP">,
  MarshallingInfoInt<LangOpts<"GPUMaxThreadsPerBlock">, "1024">,
  ShouldParseIf<hip.KeyPath>;
def fgpu_inline_threshold_EQ : Joined<["-"], "fgpu-inline-threshold=">,
  Flags<[HelpHidden]>,
  HelpText<"Inline threshold for device compilation for CUDA/HIP">;
def gpu_instrument_lib_EQ : Joined<["--"], "gpu-instrument-lib=">,
  HelpText<"Instrument device library for HIP, which is a LLVM bitcode containing "
  "__cyg_profile_func_enter and __cyg_profile_func_exit">;
def fgpu_sanitize : Flag<["-"], "fgpu-sanitize">, Group<f_Group>,
  HelpText<"Enable sanitizer for AMDGPU target">;
def fno_gpu_sanitize : Flag<["-"], "fno-gpu-sanitize">, Group<f_Group>;
def gpu_bundle_output : Flag<["--"], "gpu-bundle-output">,
  Group<f_Group>, HelpText<"Bundle output files of HIP device compilation">;
def no_gpu_bundle_output : Flag<["--"], "no-gpu-bundle-output">,
  Group<f_Group>, HelpText<"Do not bundle output files of HIP device compilation">;
def cuid_EQ : Joined<["-"], "cuid=">, Flags<[CC1Option]>,
  HelpText<"An ID for compilation unit, which should be the same for the same "
           "compilation unit but different for different compilation units. "
           "It is used to externalize device-side static variables for single "
           "source offloading languages CUDA and HIP so that they can be "
           "accessed by the host code of the same compilation unit.">,
  MarshallingInfoString<LangOpts<"CUID">>;
def fuse_cuid_EQ : Joined<["-"], "fuse-cuid=">,
  HelpText<"Method to generate ID's for compilation units for single source "
           "offloading languages CUDA and HIP: 'hash' (ID's generated by hashing "
           "file path and command line options) | 'random' (ID's generated as "
           "random numbers) | 'none' (disabled). Default is 'hash'. This option "
           "will be overridden by option '-cuid=[ID]' if it is specified." >;
def libomptarget_amdgpu_bc_path_EQ : Joined<["--"], "libomptarget-amdgpu-bc-path=">, Group<i_Group>,
  HelpText<"Path to libomptarget-amdgcn bitcode library">;
def libomptarget_amdgcn_bc_path_EQ : Joined<["--"], "libomptarget-amdgcn-bc-path=">, Group<i_Group>,
  HelpText<"Path to libomptarget-amdgcn bitcode library">, Alias<libomptarget_amdgpu_bc_path_EQ>;
def libomptarget_nvptx_bc_path_EQ : Joined<["--"], "libomptarget-nvptx-bc-path=">, Group<i_Group>,
  HelpText<"Path to libomptarget-nvptx bitcode library">;
def dD : Flag<["-"], "dD">, Group<d_Group>, Flags<[CC1Option]>,
  HelpText<"Print macro definitions in -E mode in addition to normal output">;
def dI : Flag<["-"], "dI">, Group<d_Group>, Flags<[CC1Option]>,
  HelpText<"Print include directives in -E mode in addition to normal output">,
  MarshallingInfoFlag<PreprocessorOutputOpts<"ShowIncludeDirectives">>;
def dM : Flag<["-"], "dM">, Group<d_Group>, Flags<[CC1Option]>,
  HelpText<"Print macro definitions in -E mode instead of normal output">;
def dead__strip : Flag<["-"], "dead_strip">;
def dependency_file : Separate<["-"], "dependency-file">, Flags<[CC1Option]>,
  HelpText<"Filename (or -) to write dependency output to">,
  MarshallingInfoString<DependencyOutputOpts<"OutputFile">>;
def dependency_dot : Separate<["-"], "dependency-dot">, Flags<[CC1Option]>,
  HelpText<"Filename to write DOT-formatted header dependencies to">,
  MarshallingInfoString<DependencyOutputOpts<"DOTOutputFile">>;
def module_dependency_dir : Separate<["-"], "module-dependency-dir">,
  Flags<[CC1Option]>, HelpText<"Directory to dump module dependencies to">,
  MarshallingInfoString<DependencyOutputOpts<"ModuleDependencyOutputDir">>;
def dsym_dir : JoinedOrSeparate<["-"], "dsym-dir">,
  Flags<[NoXarchOption, RenderAsInput]>,
  HelpText<"Directory to output dSYM's (if any) to">, MetaVarName<"<dir>">;
def dumpmachine : Flag<["-"], "dumpmachine">;
def dumpspecs : Flag<["-"], "dumpspecs">, Flags<[Unsupported]>;
def dumpversion : Flag<["-"], "dumpversion">;
def dylib__file : Separate<["-"], "dylib_file">;
def dylinker__install__name : JoinedOrSeparate<["-"], "dylinker_install_name">;
def dylinker : Flag<["-"], "dylinker">;
def dynamiclib : Flag<["-"], "dynamiclib">;
def dynamic : Flag<["-"], "dynamic">, Flags<[NoArgumentUnused]>;
def d_Flag : Flag<["-"], "d">, Group<d_Group>;
def d_Joined : Joined<["-"], "d">, Group<d_Group>;
def emit_ast : Flag<["-"], "emit-ast">, Flags<[CoreOption]>,
  HelpText<"Emit Clang AST files for source inputs">;
def emit_llvm : Flag<["-"], "emit-llvm">, Flags<[CC1Option, FC1Option, FlangOption]>, Group<Action_Group>,
  HelpText<"Use the LLVM representation for assembler and object files">;
def emit_interface_stubs : Flag<["-"], "emit-interface-stubs">, Flags<[CC1Option]>, Group<Action_Group>,
  HelpText<"Generate Interface Stub Files.">;
def emit_merged_ifs : Flag<["-"], "emit-merged-ifs">,
  Flags<[CC1Option]>, Group<Action_Group>,
  HelpText<"Generate Interface Stub Files, emit merged text not binary.">;
def end_no_unused_arguments : Flag<["--"], "end-no-unused-arguments">, Flags<[CoreOption]>,
  HelpText<"Start emitting warnings for unused driver arguments">;
def interface_stub_version_EQ : JoinedOrSeparate<["-"], "interface-stub-version=">, Flags<[CC1Option]>;
def exported__symbols__list : Separate<["-"], "exported_symbols_list">;
def extract_api : Flag<["-"], "extract-api">, Flags<[CC1Option]>, Group<Action_Group>,
  HelpText<"Extract API information">;
def product_name_EQ: Joined<["--"], "product-name=">, Flags<[CC1Option]>,
  MarshallingInfoString<FrontendOpts<"ProductName">>;
def e : JoinedOrSeparate<["-"], "e">, Flags<[LinkerInput]>, Group<Link_Group>;
def fmax_tokens_EQ : Joined<["-"], "fmax-tokens=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Max total number of preprocessed tokens for -Wmax-tokens.">,
  MarshallingInfoInt<LangOpts<"MaxTokens">>;
def fPIC : Flag<["-"], "fPIC">, Group<f_Group>;
def fno_PIC : Flag<["-"], "fno-PIC">, Group<f_Group>;
def fPIE : Flag<["-"], "fPIE">, Group<f_Group>;
def fno_PIE : Flag<["-"], "fno-PIE">, Group<f_Group>;
defm access_control : BoolFOption<"access-control",
  LangOpts<"AccessControl">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Disable C++ access control">,
  PosFlag<SetTrue>>;
def falign_functions : Flag<["-"], "falign-functions">, Group<f_Group>;
def falign_functions_EQ : Joined<["-"], "falign-functions=">, Group<f_Group>;
def falign_loops_EQ : Joined<["-"], "falign-loops=">, Group<f_Group>, Flags<[CC1Option]>, MetaVarName<"<N>">,
  HelpText<"N must be a power of two. Align loops to the boundary">,
  MarshallingInfoInt<CodeGenOpts<"LoopAlignment">>;
def fno_align_functions: Flag<["-"], "fno-align-functions">, Group<f_Group>;
defm allow_editor_placeholders : BoolFOption<"allow-editor-placeholders",
  LangOpts<"AllowEditorPlaceholders">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Treat editor placeholders as valid source code">,
  NegFlag<SetFalse>>;
def fallow_unsupported : Flag<["-"], "fallow-unsupported">, Group<f_Group>;
def fapple_kext : Flag<["-"], "fapple-kext">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Use Apple's kernel extensions ABI">,
  MarshallingInfoFlag<LangOpts<"AppleKext">>;
def fstrict_flex_arrays_EQ : Joined<["-"], "fstrict-flex-arrays=">,Group<f_Group>,
  MetaVarName<"<n>">, Values<"0,1,2">,
  LangOpts<"StrictFlexArrays">,
  Flags<[CC1Option]>,
  HelpText<"Enable optimizations based on the strict definition of flexible arrays">,
  MarshallingInfoInt<LangOpts<"StrictFlexArrays">>;
defm apple_pragma_pack : BoolFOption<"apple-pragma-pack",
  LangOpts<"ApplePragmaPack">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable Apple gcc-compatible #pragma pack handling">,
  NegFlag<SetFalse>>;
defm xl_pragma_pack : BoolFOption<"xl-pragma-pack",
  LangOpts<"XLPragmaPack">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable IBM XL #pragma pack handling">,
  NegFlag<SetFalse>>;
def shared_libsan : Flag<["-"], "shared-libsan">,
  HelpText<"Dynamically link the sanitizer runtime">;
def static_libsan : Flag<["-"], "static-libsan">,
  HelpText<"Statically link the sanitizer runtime">;
def : Flag<["-"], "shared-libasan">, Alias<shared_libsan>;
def fasm : Flag<["-"], "fasm">, Group<f_Group>;

def fassume_sane_operator_new : Flag<["-"], "fassume-sane-operator-new">, Group<f_Group>;
def fastcp : Flag<["-"], "fastcp">, Group<f_Group>;
def fastf : Flag<["-"], "fastf">, Group<f_Group>;
def fast : Flag<["-"], "fast">, Group<f_Group>;
def fasynchronous_unwind_tables : Flag<["-"], "fasynchronous-unwind-tables">, Group<f_Group>;

defm double_square_bracket_attributes : BoolFOption<"double-square-bracket-attributes",
  LangOpts<"DoubleSquareBracketAttributes">, Default<!strconcat(cpp11.KeyPath, "||", c2x.KeyPath)>,
  PosFlag<SetTrue, [], "Enable">, NegFlag<SetFalse, [], "Disable">,
  BothFlags<[NoXarchOption, CC1Option], " '[[]]' attributes in all C and C++ language modes">>;

defm autolink : BoolFOption<"autolink",
  CodeGenOpts<"Autolink">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Disable generation of linker directives for automatic library linking">,
  PosFlag<SetTrue>>;

// In the future this option will be supported by other offloading
// languages and accept other values such as CPU/GPU architectures,
// offload kinds and target aliases.
def offload_EQ : CommaJoined<["--"], "offload=">, Flags<[NoXarchOption]>,
  HelpText<"Specify comma-separated list of offloading target triples (CUDA and HIP only)">;

// C++ Coroutines TS
defm coroutines_ts : BoolFOption<"coroutines-ts",
  LangOpts<"Coroutines">, Default<cpp20.KeyPath>,
  PosFlag<SetTrue, [CC1Option], "Enable support for the C++ Coroutines TS">,
  NegFlag<SetFalse>>;

defm experimental_library : BoolFOption<"experimental-library",
  LangOpts<"ExperimentalLibrary">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option, CoreOption], "Control whether unstable and experimental library features are enabled. "
          "This option enables various library features that are either experimental (also known as TSes), or have been "
          "but are not stable yet in the selected Standard Library implementation. It is not recommended to use this option "
          "in production code, since neither ABI nor API stability are guaranteed. This is intended to provide a preview "
          "of features that will ship in the future for experimentation purposes">,
  NegFlag<SetFalse>>;

def fembed_offload_object_EQ : Joined<["-"], "fembed-offload-object=">,
  Group<f_Group>, Flags<[NoXarchOption, CC1Option]>,
  HelpText<"Embed Offloading device-side binary into host object file as a section.">,
  MarshallingInfoStringVector<CodeGenOpts<"OffloadObjects">>;
def fembed_bitcode_EQ : Joined<["-"], "fembed-bitcode=">,
    Group<f_Group>, Flags<[NoXarchOption, CC1Option, CC1AsOption]>, MetaVarName<"<option>">,
    HelpText<"Embed LLVM bitcode">,
    Values<"off,all,bitcode,marker">, NormalizedValuesScope<"CodeGenOptions">,
    NormalizedValues<["Embed_Off", "Embed_All", "Embed_Bitcode", "Embed_Marker"]>,
    MarshallingInfoEnum<CodeGenOpts<"EmbedBitcode">, "Embed_Off">;
def fembed_bitcode : Flag<["-"], "fembed-bitcode">, Group<f_Group>,
  Alias<fembed_bitcode_EQ>, AliasArgs<["all"]>,
  HelpText<"Embed LLVM IR bitcode as data">;
def fembed_bitcode_marker : Flag<["-"], "fembed-bitcode-marker">,
  Alias<fembed_bitcode_EQ>, AliasArgs<["marker"]>,
  HelpText<"Embed placeholder LLVM IR data as a marker">;
defm gnu_inline_asm : BoolFOption<"gnu-inline-asm",
  LangOpts<"GNUAsm">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Disable GNU style inline asm">, PosFlag<SetTrue>>;

def fprofile_sample_use : Flag<["-"], "fprofile-sample-use">, Group<f_Group>,
    Flags<[CoreOption]>;
def fno_profile_sample_use : Flag<["-"], "fno-profile-sample-use">, Group<f_Group>,
    Flags<[CoreOption]>;
def fprofile_sample_use_EQ : Joined<["-"], "fprofile-sample-use=">,
    Group<f_Group>, Flags<[NoXarchOption, CC1Option]>,
    HelpText<"Enable sample-based profile guided optimizations">,
    MarshallingInfoString<CodeGenOpts<"SampleProfileFile">>;
def fprofile_sample_accurate : Flag<["-"], "fprofile-sample-accurate">,
    Group<f_Group>, Flags<[NoXarchOption, CC1Option]>,
    HelpText<"Specifies that the sample profile is accurate">,
    DocBrief<[{Specifies that the sample profile is accurate. If the sample
               profile is accurate, callsites without profile samples are marked
               as cold. Otherwise, treat callsites without profile samples as if
               we have no profile}]>,
   MarshallingInfoFlag<CodeGenOpts<"ProfileSampleAccurate">>;
def fno_profile_sample_accurate : Flag<["-"], "fno-profile-sample-accurate">,
  Group<f_Group>, Flags<[NoXarchOption]>;
def fauto_profile : Flag<["-"], "fauto-profile">, Group<f_Group>,
    Alias<fprofile_sample_use>;
def fno_auto_profile : Flag<["-"], "fno-auto-profile">, Group<f_Group>,
    Alias<fno_profile_sample_use>;
def fauto_profile_EQ : Joined<["-"], "fauto-profile=">,
    Alias<fprofile_sample_use_EQ>;
def fauto_profile_accurate : Flag<["-"], "fauto-profile-accurate">,
    Group<f_Group>, Alias<fprofile_sample_accurate>;
def fno_auto_profile_accurate : Flag<["-"], "fno-auto-profile-accurate">,
    Group<f_Group>, Alias<fno_profile_sample_accurate>;
def fdebug_compilation_dir_EQ : Joined<["-"], "fdebug-compilation-dir=">,
    Group<f_Group>, Flags<[CC1Option, CC1AsOption, CoreOption]>,
    HelpText<"The compilation directory to embed in the debug info">,
    MarshallingInfoString<CodeGenOpts<"DebugCompilationDir">>;
def fdebug_compilation_dir : Separate<["-"], "fdebug-compilation-dir">,
    Group<f_Group>, Flags<[CC1Option, CC1AsOption, CoreOption]>,
    Alias<fdebug_compilation_dir_EQ>;
def fcoverage_compilation_dir_EQ : Joined<["-"], "fcoverage-compilation-dir=">,
    Group<f_Group>, Flags<[CC1Option, CC1AsOption, CoreOption]>,
    HelpText<"The compilation directory to embed in the coverage mapping.">,
    MarshallingInfoString<CodeGenOpts<"CoverageCompilationDir">>;
def ffile_compilation_dir_EQ : Joined<["-"], "ffile-compilation-dir=">, Group<f_Group>,
    Flags<[CoreOption]>,
    HelpText<"The compilation directory to embed in the debug info and coverage mapping.">;
defm debug_info_for_profiling : BoolFOption<"debug-info-for-profiling",
  CodeGenOpts<"DebugInfoForProfiling">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Emit extra debug info to make sample profile more accurate">,
  NegFlag<SetFalse>>;
def fprofile_instr_generate : Flag<["-"], "fprofile-instr-generate">,
    Group<f_Group>, Flags<[CoreOption]>,
    HelpText<"Generate instrumented code to collect execution counts into default.profraw file (overridden by '=' form of option or LLVM_PROFILE_FILE env var)">;
def fprofile_instr_generate_EQ : Joined<["-"], "fprofile-instr-generate=">,
    Group<f_Group>, Flags<[CoreOption]>, MetaVarName<"<file>">,
    HelpText<"Generate instrumented code to collect execution counts into <file> (overridden by LLVM_PROFILE_FILE env var)">;
def fprofile_instr_use : Flag<["-"], "fprofile-instr-use">, Group<f_Group>,
    Flags<[CoreOption]>;
def fprofile_instr_use_EQ : Joined<["-"], "fprofile-instr-use=">,
    Group<f_Group>, Flags<[CoreOption]>,
    HelpText<"Use instrumentation data for profile-guided optimization">;
def fprofile_remapping_file_EQ : Joined<["-"], "fprofile-remapping-file=">,
    Group<f_Group>, Flags<[CC1Option, CoreOption]>, MetaVarName<"<file>">,
    HelpText<"Use the remappings described in <file> to match the profile data against names in the program">,
    MarshallingInfoString<CodeGenOpts<"ProfileRemappingFile">>;
defm coverage_mapping : BoolFOption<"coverage-mapping",
  CodeGenOpts<"CoverageMapping">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Generate coverage mapping to enable code coverage analysis">,
  NegFlag<SetFalse, [], "Disable code coverage analysis">, BothFlags<[CoreOption]>>;
def fprofile_generate : Flag<["-"], "fprofile-generate">,
    Group<f_Group>, Flags<[CoreOption]>,
    HelpText<"Generate instrumented code to collect execution counts into default.profraw (overridden by LLVM_PROFILE_FILE env var)">;
def fprofile_generate_EQ : Joined<["-"], "fprofile-generate=">,
    Group<f_Group>, Flags<[CoreOption]>, MetaVarName<"<directory>">,
    HelpText<"Generate instrumented code to collect execution counts into <directory>/default.profraw (overridden by LLVM_PROFILE_FILE env var)">;
def fcs_profile_generate : Flag<["-"], "fcs-profile-generate">,
    Group<f_Group>, Flags<[CoreOption]>,
    HelpText<"Generate instrumented code to collect context sensitive execution counts into default.profraw (overridden by LLVM_PROFILE_FILE env var)">;
def fcs_profile_generate_EQ : Joined<["-"], "fcs-profile-generate=">,
    Group<f_Group>, Flags<[CoreOption]>, MetaVarName<"<directory>">,
    HelpText<"Generate instrumented code to collect context sensitive execution counts into <directory>/default.profraw (overridden by LLVM_PROFILE_FILE env var)">;
def fprofile_use : Flag<["-"], "fprofile-use">, Group<f_Group>,
    Flags<[CoreOption]>, Alias<fprofile_instr_use>;
def fprofile_use_EQ : Joined<["-"], "fprofile-use=">,
    Group<f_Group>, Flags<[NoXarchOption, CoreOption]>,
    MetaVarName<"<pathname>">,
    HelpText<"Use instrumentation data for profile-guided optimization. If pathname is a directory, it reads from <pathname>/default.profdata. Otherwise, it reads from file <pathname>.">;
def fno_profile_instr_generate : Flag<["-"], "fno-profile-instr-generate">,
    Group<f_Group>, Flags<[CoreOption]>,
    HelpText<"Disable generation of profile instrumentation.">;
def fno_profile_generate : Flag<["-"], "fno-profile-generate">,
    Group<f_Group>, Flags<[CoreOption]>,
    HelpText<"Disable generation of profile instrumentation.">;
def fno_profile_instr_use : Flag<["-"], "fno-profile-instr-use">,
    Group<f_Group>, Flags<[CoreOption]>,
    HelpText<"Disable using instrumentation data for profile-guided optimization">;
def fno_profile_use : Flag<["-"], "fno-profile-use">,
    Alias<fno_profile_instr_use>;
defm profile_arcs : BoolFOption<"profile-arcs",
  CodeGenOpts<"EmitGcovArcs">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option, LinkOption]>, NegFlag<SetFalse>>;
defm test_coverage : BoolFOption<"test-coverage",
  CodeGenOpts<"EmitGcovNotes">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option]>, NegFlag<SetFalse>>;
def fprofile_filter_files_EQ : Joined<["-"], "fprofile-filter-files=">,
    Group<f_Group>, Flags<[CC1Option, CoreOption]>,
    HelpText<"Instrument only functions from files where names match any regex separated by a semi-colon">,
    MarshallingInfoString<CodeGenOpts<"ProfileFilterFiles">>,
    ShouldParseIf<!strconcat(fprofile_arcs.KeyPath, "||", ftest_coverage.KeyPath)>;
def fprofile_exclude_files_EQ : Joined<["-"], "fprofile-exclude-files=">,
    Group<f_Group>, Flags<[CC1Option, CoreOption]>,
    HelpText<"Instrument only functions from files where names don't match all the regexes separated by a semi-colon">,
    MarshallingInfoString<CodeGenOpts<"ProfileExcludeFiles">>,
    ShouldParseIf<!strconcat(fprofile_arcs.KeyPath, "||", ftest_coverage.KeyPath)>;
def fprofile_update_EQ : Joined<["-"], "fprofile-update=">,
    Group<f_Group>, Flags<[CC1Option, CoreOption]>, Values<"atomic,prefer-atomic,single">,
    MetaVarName<"<method>">, HelpText<"Set update method of profile counters">,
    MarshallingInfoFlag<CodeGenOpts<"AtomicProfileUpdate">>;
defm pseudo_probe_for_profiling : BoolFOption<"pseudo-probe-for-profiling",
  CodeGenOpts<"PseudoProbeForProfiling">, DefaultFalse,
  PosFlag<SetTrue, [], "Emit">, NegFlag<SetFalse, [], "Do not emit">,
  BothFlags<[NoXarchOption, CC1Option], " pseudo probes for sample profiling">>;
def forder_file_instrumentation : Flag<["-"], "forder-file-instrumentation">,
    Group<f_Group>, Flags<[CC1Option, CoreOption]>,
    HelpText<"Generate instrumented code to collect order file into default.profraw file (overridden by '=' form of option or LLVM_PROFILE_FILE env var)">;
def fprofile_list_EQ : Joined<["-"], "fprofile-list=">,
    Group<f_Group>, Flags<[CC1Option, CoreOption]>,
    HelpText<"Filename defining the list of functions/files to instrument">,
    MarshallingInfoStringVector<LangOpts<"ProfileListFiles">>;
def fprofile_function_groups : Joined<["-"], "fprofile-function-groups=">,
  Group<f_Group>, Flags<[CC1Option]>, MetaVarName<"<N>">,
  HelpText<"Partition functions into N groups and select only functions in group i to be instrumented using -fprofile-selected-function-group">,
  MarshallingInfoInt<CodeGenOpts<"ProfileTotalFunctionGroups">, "1">;
def fprofile_selected_function_group :
  Joined<["-"], "fprofile-selected-function-group=">, Group<f_Group>,
  Flags<[CC1Option]>, MetaVarName<"<i>">,
  HelpText<"Partition functions into N groups using -fprofile-function-groups and select only functions in group i to be instrumented. The valid range is 0 to N-1 inclusive">,
  MarshallingInfoInt<CodeGenOpts<"ProfileSelectedFunctionGroup">>;
def fswift_async_fp_EQ : Joined<["-"], "fswift-async-fp=">,
    Group<f_Group>, Flags<[CC1Option, CC1AsOption, CoreOption]>, MetaVarName<"<option>">,
    HelpText<"Control emission of Swift async extended frame info">,
    Values<"auto,always,never">,
    NormalizedValuesScope<"CodeGenOptions::SwiftAsyncFramePointerKind">,
    NormalizedValues<["Auto", "Always", "Never"]>,
    MarshallingInfoEnum<CodeGenOpts<"SwiftAsyncFramePointer">, "Always">;

defm addrsig : BoolFOption<"addrsig",
  CodeGenOpts<"Addrsig">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Emit">, NegFlag<SetFalse, [], "Don't emit">,
  BothFlags<[CoreOption], " an address-significance table">>;
defm blocks : OptInCC1FFlag<"blocks", "Enable the 'blocks' language feature", "", "", [CoreOption]>;
def fbootclasspath_EQ : Joined<["-"], "fbootclasspath=">, Group<f_Group>;
defm borland_extensions : BoolFOption<"borland-extensions",
  LangOpts<"Borland">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Accept non-standard constructs supported by the Borland compiler">,
  NegFlag<SetFalse>>;
def fbuiltin : Flag<["-"], "fbuiltin">, Group<f_Group>, Flags<[CoreOption]>;
def fbuiltin_module_map : Flag <["-"], "fbuiltin-module-map">, Group<f_Group>,
  Flags<[NoXarchOption]>, HelpText<"Load the clang builtins module map file.">;
defm caret_diagnostics : BoolFOption<"caret-diagnostics",
  DiagnosticOpts<"ShowCarets">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option]>, PosFlag<SetTrue>>;
def fclang_abi_compat_EQ : Joined<["-"], "fclang-abi-compat=">, Group<f_clang_Group>,
  Flags<[CC1Option]>, MetaVarName<"<version>">, Values<"<major>.<minor>,latest">,
  HelpText<"Attempt to match the ABI of Clang <version>">;
def fclasspath_EQ : Joined<["-"], "fclasspath=">, Group<f_Group>;
def fcolor_diagnostics : Flag<["-"], "fcolor-diagnostics">, Group<f_Group>,
  Flags<[CoreOption, CC1Option, FlangOption, FC1Option]>,
  HelpText<"Enable colors in diagnostics">;
def fno_color_diagnostics : Flag<["-"], "fno-color-diagnostics">, Group<f_Group>,
  Flags<[CoreOption, FlangOption]>, HelpText<"Disable colors in diagnostics">;
def : Flag<["-"], "fdiagnostics-color">, Group<f_Group>, Flags<[CoreOption]>, Alias<fcolor_diagnostics>;
def : Flag<["-"], "fno-diagnostics-color">, Group<f_Group>, Flags<[CoreOption]>, Alias<fno_color_diagnostics>;
def fdiagnostics_color_EQ : Joined<["-"], "fdiagnostics-color=">, Group<f_Group>;
def fansi_escape_codes : Flag<["-"], "fansi-escape-codes">, Group<f_Group>,
  Flags<[CoreOption, CC1Option]>, HelpText<"Use ANSI escape codes for diagnostics">,
  MarshallingInfoFlag<DiagnosticOpts<"UseANSIEscapeCodes">>;
def fcomment_block_commands : CommaJoined<["-"], "fcomment-block-commands=">, Group<f_clang_Group>, Flags<[CC1Option]>,
  HelpText<"Treat each comma separated argument in <arg> as a documentation comment block command">,
  MetaVarName<"<arg>">, MarshallingInfoStringVector<LangOpts<"CommentOpts.BlockCommandNames">>;
def fparse_all_comments : Flag<["-"], "fparse-all-comments">, Group<f_clang_Group>, Flags<[CC1Option]>,
  MarshallingInfoFlag<LangOpts<"CommentOpts.ParseAllComments">>;
def frecord_command_line : Flag<["-"], "frecord-command-line">,
  Group<f_clang_Group>;
def fno_record_command_line : Flag<["-"], "fno-record-command-line">,
  Group<f_clang_Group>;
def : Flag<["-"], "frecord-gcc-switches">, Alias<frecord_command_line>;
def : Flag<["-"], "fno-record-gcc-switches">, Alias<fno_record_command_line>;
def fcommon : Flag<["-"], "fcommon">, Group<f_Group>,
  Flags<[CoreOption, CC1Option]>, HelpText<"Place uninitialized global variables in a common block">,
  MarshallingInfoNegativeFlag<CodeGenOpts<"NoCommon">>;
def fcompile_resource_EQ : Joined<["-"], "fcompile-resource=">, Group<f_Group>;
defm complete_member_pointers : BoolOption<"f", "complete-member-pointers",
  LangOpts<"CompleteMemberPointers">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Require">, NegFlag<SetFalse, [], "Do not require">,
  BothFlags<[CoreOption], " member pointer base types to be complete if they"
            " would be significant under the Microsoft ABI">>,
  Group<f_clang_Group>;
def fcf_runtime_abi_EQ : Joined<["-"], "fcf-runtime-abi=">, Group<f_Group>,
    Flags<[CC1Option]>, Values<"unspecified,standalone,objc,swift,swift-5.0,swift-4.2,swift-4.1">,
    NormalizedValuesScope<"LangOptions::CoreFoundationABI">,
    NormalizedValues<["ObjectiveC", "ObjectiveC", "ObjectiveC", "Swift5_0", "Swift5_0", "Swift4_2", "Swift4_1"]>,
    MarshallingInfoEnum<LangOpts<"CFRuntime">, "ObjectiveC">;
defm constant_cfstrings : BoolFOption<"constant-cfstrings",
  LangOpts<"NoConstantCFStrings">, DefaultFalse,
  NegFlag<SetTrue, [CC1Option], "Disable creation of CodeFoundation-type constant strings">,
  PosFlag<SetFalse>>;
def fconstant_string_class_EQ : Joined<["-"], "fconstant-string-class=">, Group<f_Group>;
def fconstexpr_depth_EQ : Joined<["-"], "fconstexpr-depth=">, Group<f_Group>;
def fconstexpr_steps_EQ : Joined<["-"], "fconstexpr-steps=">, Group<f_Group>;
def fexperimental_new_constant_interpreter : Flag<["-"], "fexperimental-new-constant-interpreter">, Group<f_Group>,
  HelpText<"Enable the experimental new constant interpreter">, Flags<[CC1Option]>,
  MarshallingInfoFlag<LangOpts<"EnableNewConstInterp">>;
def fconstexpr_backtrace_limit_EQ : Joined<["-"], "fconstexpr-backtrace-limit=">,
                                    Group<f_Group>;
def fno_crash_diagnostics : Flag<["-"], "fno-crash-diagnostics">, Group<f_clang_Group>, Flags<[NoArgumentUnused, CoreOption]>,
  Alias<gen_reproducer_eq>, AliasArgs<["off"]>,
  HelpText<"Disable auto-generation of preprocessed source files and a script for reproduction during a clang crash">;
def fcrash_diagnostics_dir : Joined<["-"], "fcrash-diagnostics-dir=">,
  Group<f_clang_Group>, Flags<[NoArgumentUnused, CoreOption]>,
  HelpText<"Put crash-report files in <dir>">, MetaVarName<"<dir>">;
def fcreate_profile : Flag<["-"], "fcreate-profile">, Group<f_Group>;
defm cxx_exceptions: BoolFOption<"cxx-exceptions",
  LangOpts<"CXXExceptions">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable C++ exceptions">, NegFlag<SetFalse>>;
defm async_exceptions: BoolFOption<"async-exceptions",
  LangOpts<"EHAsynch">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable EH Asynchronous exceptions">, NegFlag<SetFalse>>;
defm cxx_modules : BoolFOption<"cxx-modules",
  LangOpts<"CPlusPlusModules">, Default<cpp20.KeyPath>,
  NegFlag<SetFalse, [CC1Option], "Disable">, PosFlag<SetTrue, [], "Enable">,
  BothFlags<[NoXarchOption], " modules for C++">>,
  ShouldParseIf<cplusplus.KeyPath>;
def fdebug_pass_arguments : Flag<["-"], "fdebug-pass-arguments">, Group<f_Group>;
def fdebug_pass_structure : Flag<["-"], "fdebug-pass-structure">, Group<f_Group>;
def fdepfile_entry : Joined<["-"], "fdepfile-entry=">,
    Group<f_clang_Group>, Flags<[CC1Option]>;
def fdiagnostics_fixit_info : Flag<["-"], "fdiagnostics-fixit-info">, Group<f_clang_Group>;
def fno_diagnostics_fixit_info : Flag<["-"], "fno-diagnostics-fixit-info">, Group<f_Group>,
  Flags<[CC1Option]>, HelpText<"Do not include fixit information in diagnostics">,
  MarshallingInfoNegativeFlag<DiagnosticOpts<"ShowFixits">>;
def fdiagnostics_parseable_fixits : Flag<["-"], "fdiagnostics-parseable-fixits">, Group<f_clang_Group>,
    Flags<[CoreOption, CC1Option]>, HelpText<"Print fix-its in machine parseable form">,
    MarshallingInfoFlag<DiagnosticOpts<"ShowParseableFixits">>;
def fdiagnostics_print_source_range_info : Flag<["-"], "fdiagnostics-print-source-range-info">,
    Group<f_clang_Group>,  Flags<[CC1Option]>,
    HelpText<"Print source range spans in numeric form">,
    MarshallingInfoFlag<DiagnosticOpts<"ShowSourceRanges">>;
defm diagnostics_show_hotness : BoolFOption<"diagnostics-show-hotness",
  CodeGenOpts<"DiagnosticsWithHotness">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable profile hotness information in diagnostic line">,
  NegFlag<SetFalse>>;
def fdiagnostics_hotness_threshold_EQ : Joined<["-"], "fdiagnostics-hotness-threshold=">,
    Group<f_Group>, Flags<[CC1Option]>, MetaVarName<"<value>">,
    HelpText<"Prevent optimization remarks from being output if they do not have at least this profile count. "
    "Use 'auto' to apply the threshold from profile summary">;
def fdiagnostics_misexpect_tolerance_EQ : Joined<["-"], "fdiagnostics-misexpect-tolerance=">,
    Group<f_Group>, Flags<[CC1Option]>, MetaVarName<"<value>">,
    HelpText<"Prevent misexpect diagnostics from being output if the profile counts are within N% of the expected. ">;
defm diagnostics_show_option : BoolFOption<"diagnostics-show-option",
    DiagnosticOpts<"ShowOptionNames">, DefaultTrue,
    NegFlag<SetFalse, [CC1Option]>, PosFlag<SetTrue, [], "Print option name with mappable diagnostics">>;
defm diagnostics_show_note_include_stack : BoolFOption<"diagnostics-show-note-include-stack",
    DiagnosticOpts<"ShowNoteIncludeStack">, DefaultFalse,
    PosFlag<SetTrue, [], "Display include stacks for diagnostic notes">,
    NegFlag<SetFalse>, BothFlags<[CC1Option]>>;
def fdiagnostics_format_EQ : Joined<["-"], "fdiagnostics-format=">, Group<f_clang_Group>;
def fdiagnostics_show_category_EQ : Joined<["-"], "fdiagnostics-show-category=">, Group<f_clang_Group>;
def fdiagnostics_show_template_tree : Flag<["-"], "fdiagnostics-show-template-tree">,
    Group<f_Group>, Flags<[CC1Option]>,
    HelpText<"Print a template comparison tree for differing templates">,
    MarshallingInfoFlag<DiagnosticOpts<"ShowTemplateTree">>;
def fdiscard_value_names : Flag<["-"], "fdiscard-value-names">, Group<f_clang_Group>,
  HelpText<"Discard value names in LLVM IR">, Flags<[NoXarchOption]>;
def fno_discard_value_names : Flag<["-"], "fno-discard-value-names">, Group<f_clang_Group>,
  HelpText<"Do not discard value names in LLVM IR">, Flags<[NoXarchOption]>;
defm dollars_in_identifiers : BoolFOption<"dollars-in-identifiers",
  LangOpts<"DollarIdents">, Default<!strconcat("!", asm_preprocessor.KeyPath)>,
  PosFlag<SetTrue, [], "Allow">, NegFlag<SetFalse, [], "Disallow">,
  BothFlags<[CC1Option], " '$' in identifiers">>;
def fdwarf2_cfi_asm : Flag<["-"], "fdwarf2-cfi-asm">, Group<clang_ignored_f_Group>;
def fno_dwarf2_cfi_asm : Flag<["-"], "fno-dwarf2-cfi-asm">, Group<clang_ignored_f_Group>;
defm dwarf_directory_asm : BoolFOption<"dwarf-directory-asm",
  CodeGenOpts<"NoDwarfDirectoryAsm">, DefaultFalse,
  NegFlag<SetTrue, [CC1Option]>, PosFlag<SetFalse>>;
defm elide_constructors : BoolFOption<"elide-constructors",
  LangOpts<"ElideConstructors">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Disable C++ copy constructor elision">,
  PosFlag<SetTrue>>;
def fno_elide_type : Flag<["-"], "fno-elide-type">, Group<f_Group>,
    Flags<[CC1Option]>,
    HelpText<"Do not elide types when printing diagnostics">,
    MarshallingInfoNegativeFlag<DiagnosticOpts<"ElideType">>;
def feliminate_unused_debug_symbols : Flag<["-"], "feliminate-unused-debug-symbols">, Group<f_Group>;
defm eliminate_unused_debug_types : OptOutCC1FFlag<"eliminate-unused-debug-types",
  "Do not emit ", "Emit ", " debug info for defined but unused types">;
def femit_all_decls : Flag<["-"], "femit-all-decls">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Emit all declarations, even if unused">,
  MarshallingInfoFlag<LangOpts<"EmitAllDecls">>;
defm emulated_tls : BoolFOption<"emulated-tls",
  CodeGenOpts<"EmulatedTLS">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Use emutls functions to access thread_local variables">,
  NegFlag<SetFalse>, BothFlags<[CC1Option]>>;
def fencoding_EQ : Joined<["-"], "fencoding=">, Group<f_Group>;
def ferror_limit_EQ : Joined<["-"], "ferror-limit=">, Group<f_Group>, Flags<[CoreOption]>;
defm exceptions : BoolFOption<"exceptions",
  LangOpts<"Exceptions">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable">, NegFlag<SetFalse, [], "Disable">,
  BothFlags<[], " support for exception handling">>;
def fdwarf_exceptions : Flag<["-"], "fdwarf-exceptions">, Group<f_Group>,
  HelpText<"Use DWARF style exceptions">;
def fsjlj_exceptions : Flag<["-"], "fsjlj-exceptions">, Group<f_Group>,
  HelpText<"Use SjLj style exceptions">;
def fseh_exceptions : Flag<["-"], "fseh-exceptions">, Group<f_Group>,
  HelpText<"Use SEH style exceptions">;
def fwasm_exceptions : Flag<["-"], "fwasm-exceptions">, Group<f_Group>,
  HelpText<"Use WebAssembly style exceptions">;
def exception_model : Separate<["-"], "exception-model">,
  Flags<[CC1Option, NoDriverOption]>, HelpText<"The exception model">,
  Values<"dwarf,sjlj,seh,wasm">,
  NormalizedValuesScope<"LangOptions::ExceptionHandlingKind">,
  NormalizedValues<["DwarfCFI", "SjLj", "WinEH", "Wasm"]>,
  MarshallingInfoEnum<LangOpts<"ExceptionHandling">, "None">;
def exception_model_EQ : Joined<["-"], "exception-model=">,
  Flags<[CC1Option, NoDriverOption]>, Alias<exception_model>;
def fignore_exceptions : Flag<["-"], "fignore-exceptions">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable support for ignoring exception handling constructs">,
  MarshallingInfoFlag<LangOpts<"IgnoreExceptions">>;
def fexcess_precision_EQ : Joined<["-"], "fexcess-precision=">,
    Group<clang_ignored_gcc_optimization_f_Group>;
def : Flag<["-"], "fexpensive-optimizations">, Group<clang_ignored_gcc_optimization_f_Group>;
def : Flag<["-"], "fno-expensive-optimizations">, Group<clang_ignored_gcc_optimization_f_Group>;
def fextdirs_EQ : Joined<["-"], "fextdirs=">, Group<f_Group>;
def : Flag<["-"], "fdefer-pop">, Group<clang_ignored_gcc_optimization_f_Group>;
def : Flag<["-"], "fno-defer-pop">, Group<clang_ignored_gcc_optimization_f_Group>;
def : Flag<["-"], "fextended-identifiers">, Group<clang_ignored_f_Group>;
def : Flag<["-"], "fno-extended-identifiers">, Group<f_Group>, Flags<[Unsupported]>;
def fhosted : Flag<["-"], "fhosted">, Group<f_Group>;
def fdenormal_fp_math_EQ : Joined<["-"], "fdenormal-fp-math=">, Group<f_Group>, Flags<[CC1Option]>;
def ffile_reproducible : Flag<["-"], "ffile-reproducible">, Group<f_Group>,
  Flags<[CoreOption, CC1Option]>,
  HelpText<"Use the target's platform-specific path separator character when "
           "expanding the __FILE__ macro">;
def fno_file_reproducible : Flag<["-"], "fno-file-reproducible">,
  Group<f_Group>, Flags<[CoreOption, CC1Option]>,
  HelpText<"Use the host's platform-specific path separator character when "
           "expanding the __FILE__ macro">;
def ffp_eval_method_EQ : Joined<["-"], "ffp-eval-method=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Specifies the evaluation method to use for floating-point arithmetic.">,
  Values<"source,double,extended">, NormalizedValuesScope<"LangOptions">,
  NormalizedValues<["FEM_Source", "FEM_Double", "FEM_Extended"]>,
  MarshallingInfoEnum<LangOpts<"FPEvalMethod">, "FEM_UnsetOnCommandLine">;
def ffp_model_EQ : Joined<["-"], "ffp-model=">, Group<f_Group>, Flags<[NoXarchOption]>,
  HelpText<"Controls the semantics of floating-point calculations.">;
def ffp_exception_behavior_EQ : Joined<["-"], "ffp-exception-behavior=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Specifies the exception behavior of floating-point operations.">,
  Values<"ignore,maytrap,strict">, NormalizedValuesScope<"LangOptions">,
  NormalizedValues<["FPE_Ignore", "FPE_MayTrap", "FPE_Strict"]>,
  MarshallingInfoEnum<LangOpts<"FPExceptionMode">, "FPE_Default">;
defm fast_math : BoolFOption<"fast-math",
  LangOpts<"FastMath">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Allow aggressive, lossy floating-point optimizations",
          [cl_fast_relaxed_math.KeyPath]>,
  NegFlag<SetFalse>>;
def menable_unsafe_fp_math : Flag<["-"], "menable-unsafe-fp-math">, Flags<[CC1Option]>,
  HelpText<"Allow unsafe floating-point math optimizations which may decrease precision">,
  MarshallingInfoFlag<LangOpts<"UnsafeFPMath">>,
  ImpliedByAnyOf<[cl_unsafe_math_optimizations.KeyPath, ffast_math.KeyPath]>;
defm math_errno : BoolFOption<"math-errno",
  LangOpts<"MathErrno">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Require math functions to indicate errors by setting errno">,
  NegFlag<SetFalse>>,
  ShouldParseIf<!strconcat("!", open_cl.KeyPath)>;
def fextend_args_EQ : Joined<["-"], "fextend-arguments=">, Group<f_Group>,
  Flags<[CC1Option, NoArgumentUnused]>,
  HelpText<"Controls how scalar integer arguments are extended in calls "
           "to unprototyped and varargs functions">,
  Values<"32,64">,
  NormalizedValues<["ExtendTo32", "ExtendTo64"]>,
  NormalizedValuesScope<"LangOptions::ExtendArgsKind">,
  MarshallingInfoEnum<LangOpts<"ExtendIntArgs">,"ExtendTo32">;
def fbracket_depth_EQ : Joined<["-"], "fbracket-depth=">, Group<f_Group>, Flags<[CoreOption]>;
def fsignaling_math : Flag<["-"], "fsignaling-math">, Group<f_Group>;
def fno_signaling_math : Flag<["-"], "fno-signaling-math">, Group<f_Group>;
defm jump_tables : BoolFOption<"jump-tables",
  CodeGenOpts<"NoUseJumpTables">, DefaultFalse,
  NegFlag<SetTrue, [CC1Option], "Do not use">, PosFlag<SetFalse, [], "Use">,
  BothFlags<[], " jump tables for lowering switches">>;
defm force_enable_int128 : BoolFOption<"force-enable-int128",
  TargetOpts<"ForceEnableInt128">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable">, NegFlag<SetFalse, [], "Disable">,
  BothFlags<[], " support for int128_t type">>;
defm keep_static_consts : BoolFOption<"keep-static-consts",
  CodeGenOpts<"KeepStaticConsts">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Keep">, NegFlag<SetFalse, [], "Don't keep">,
  BothFlags<[NoXarchOption], " static const variables if unused">>;
defm fixed_point : BoolFOption<"fixed-point",
  LangOpts<"FixedPoint">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable">, NegFlag<SetFalse, [], "Disable">,
  BothFlags<[], " fixed point types">>, ShouldParseIf<!strconcat("!", cplusplus.KeyPath)>;
defm cxx_static_destructors : BoolFOption<"c++-static-destructors",
  LangOpts<"RegisterStaticDestructors">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Disable C++ static destructor registration">,
  PosFlag<SetTrue>>;
def fsymbol_partition_EQ : Joined<["-"], "fsymbol-partition=">, Group<f_Group>,
  Flags<[CC1Option]>, MarshallingInfoString<CodeGenOpts<"SymbolPartition">>;

defm memory_profile : OptInCC1FFlag<"memory-profile", "Enable", "Disable", " heap memory profiling">;
def fmemory_profile_EQ : Joined<["-"], "fmemory-profile=">,
    Group<f_Group>, Flags<[CC1Option]>, MetaVarName<"<directory>">,
    HelpText<"Enable heap memory profiling and dump results into <directory>">;

// Begin sanitizer flags. These should all be core options exposed in all driver
// modes.
let Flags = [CC1Option, CoreOption] in {

def fsanitize_EQ : CommaJoined<["-"], "fsanitize=">, Group<f_clang_Group>,
                   MetaVarName<"<check>">,
                   HelpText<"Turn on runtime checks for various forms of undefined "
                            "or suspicious behavior. See user manual for available checks">;
def fno_sanitize_EQ : CommaJoined<["-"], "fno-sanitize=">, Group<f_clang_Group>,
                      Flags<[CoreOption, NoXarchOption]>;

def fsanitize_ignorelist_EQ : Joined<["-"], "fsanitize-ignorelist=">,
  Group<f_clang_Group>, HelpText<"Path to ignorelist file for sanitizers">;
def : Joined<["-"], "fsanitize-blacklist=">,
  Group<f_clang_Group>, Flags<[HelpHidden]>, Alias<fsanitize_ignorelist_EQ>,
  HelpText<"Alias for -fsanitize-ignorelist=">;

def fsanitize_system_ignorelist_EQ : Joined<["-"], "fsanitize-system-ignorelist=">,
  HelpText<"Path to system ignorelist file for sanitizers">, Flags<[CC1Option]>;
def : Joined<["-"], "fsanitize-system-blacklist=">,
  HelpText<"Alias for -fsanitize-system-ignorelist=">,
  Flags<[CC1Option, HelpHidden]>, Alias<fsanitize_system_ignorelist_EQ>;

def fno_sanitize_ignorelist : Flag<["-"], "fno-sanitize-ignorelist">,
  Group<f_clang_Group>, HelpText<"Don't use ignorelist file for sanitizers">;
def : Flag<["-"], "fno-sanitize-blacklist">,
  Group<f_clang_Group>, Flags<[HelpHidden]>, Alias<fno_sanitize_ignorelist>;

def fsanitize_coverage : CommaJoined<["-"], "fsanitize-coverage=">,
  Group<f_clang_Group>,
  HelpText<"Specify the type of coverage instrumentation for Sanitizers">;
def fno_sanitize_coverage : CommaJoined<["-"], "fno-sanitize-coverage=">,
  Group<f_clang_Group>, Flags<[CoreOption, NoXarchOption]>,
  HelpText<"Disable features of coverage instrumentation for Sanitizers">,
  Values<"func,bb,edge,indirect-calls,trace-bb,trace-cmp,trace-div,trace-gep,"
         "8bit-counters,trace-pc,trace-pc-guard,no-prune,inline-8bit-counters,"
         "inline-bool-flag">;
def fsanitize_coverage_allowlist : Joined<["-"], "fsanitize-coverage-allowlist=">,
    Group<f_clang_Group>, Flags<[CoreOption, NoXarchOption]>,
    HelpText<"Restrict sanitizer coverage instrumentation exclusively to modules and functions that match the provided special case list, except the blocked ones">,
    MarshallingInfoStringVector<CodeGenOpts<"SanitizeCoverageAllowlistFiles">>;
def : Joined<["-"], "fsanitize-coverage-whitelist=">,
  Group<f_clang_Group>, Flags<[CoreOption, HelpHidden]>, Alias<fsanitize_coverage_allowlist>,
  HelpText<"Deprecated, use -fsanitize-coverage-allowlist= instead">;
def fsanitize_coverage_ignorelist : Joined<["-"], "fsanitize-coverage-ignorelist=">,
    Group<f_clang_Group>, Flags<[CoreOption, NoXarchOption]>,
    HelpText<"Disable sanitizer coverage instrumentation for modules and functions "
             "that match the provided special case list, even the allowed ones">,
    MarshallingInfoStringVector<CodeGenOpts<"SanitizeCoverageIgnorelistFiles">>;
def : Joined<["-"], "fsanitize-coverage-blacklist=">,
  Group<f_clang_Group>, Flags<[CoreOption, HelpHidden]>,
  Alias<fsanitize_coverage_ignorelist>,
  HelpText<"Deprecated, use -fsanitize-coverage-ignorelist= instead">;
def fsanitize_memory_track_origins_EQ : Joined<["-"], "fsanitize-memory-track-origins=">,
                                        Group<f_clang_Group>,
                                        HelpText<"Enable origins tracking in MemorySanitizer">,
                                        MarshallingInfoInt<CodeGenOpts<"SanitizeMemoryTrackOrigins">>;
def fsanitize_memory_track_origins : Flag<["-"], "fsanitize-memory-track-origins">,
                                     Group<f_clang_Group>,
                                     HelpText<"Enable origins tracking in MemorySanitizer">;
def fno_sanitize_memory_track_origins : Flag<["-"], "fno-sanitize-memory-track-origins">,
                                        Group<f_clang_Group>,
                                        Flags<[CoreOption, NoXarchOption]>,
                                        HelpText<"Disable origins tracking in MemorySanitizer">;
def fsanitize_address_outline_instrumentation : Flag<["-"], "fsanitize-address-outline-instrumentation">,
                                                Group<f_clang_Group>,
                                                HelpText<"Always generate function calls for address sanitizer instrumentation">;
def fno_sanitize_address_outline_instrumentation : Flag<["-"], "fno-sanitize-address-outline-instrumentation">,
                                                   Group<f_clang_Group>,
                                                   HelpText<"Use default code inlining logic for the address sanitizer">;
def fsanitize_memtag_mode_EQ : Joined<["-"], "fsanitize-memtag-mode=">,
                                        Group<f_clang_Group>,
                                        HelpText<"Set default MTE mode to 'sync' (default) or 'async'">;
def fsanitize_hwaddress_experimental_aliasing
  : Flag<["-"], "fsanitize-hwaddress-experimental-aliasing">,
    Group<f_clang_Group>,
    HelpText<"Enable aliasing mode in HWAddressSanitizer">;
def fno_sanitize_hwaddress_experimental_aliasing
  : Flag<["-"], "fno-sanitize-hwaddress-experimental-aliasing">,
    Group<f_clang_Group>, Flags<[CoreOption, NoXarchOption]>,
    HelpText<"Disable aliasing mode in HWAddressSanitizer">;
defm sanitize_memory_use_after_dtor : BoolOption<"f", "sanitize-memory-use-after-dtor",
  CodeGenOpts<"SanitizeMemoryUseAfterDtor">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable">, NegFlag<SetFalse, [], "Disable">,
  BothFlags<[], " use-after-destroy detection in MemorySanitizer">>,
  Group<f_clang_Group>;
def fsanitize_address_field_padding : Joined<["-"], "fsanitize-address-field-padding=">,
                                        Group<f_clang_Group>,
                                        HelpText<"Level of field padding for AddressSanitizer">,
                                        MarshallingInfoInt<LangOpts<"SanitizeAddressFieldPadding">>;
defm sanitize_address_use_after_scope : BoolOption<"f", "sanitize-address-use-after-scope",
  CodeGenOpts<"SanitizeAddressUseAfterScope">, DefaultFalse,
  PosFlag<SetTrue, [], "Enable">, NegFlag<SetFalse, [CoreOption, NoXarchOption], "Disable">,
  BothFlags<[], " use-after-scope detection in AddressSanitizer">>,
  Group<f_clang_Group>;
def sanitize_address_use_after_return_EQ
  : Joined<["-"], "fsanitize-address-use-after-return=">,
    MetaVarName<"<mode>">,
    Flags<[CC1Option]>,
    HelpText<"Select the mode of detecting stack use-after-return in AddressSanitizer">,
    Group<f_clang_Group>,
    Values<"never,runtime,always">,
    NormalizedValuesScope<"llvm::AsanDetectStackUseAfterReturnMode">,
    NormalizedValues<["Never", "Runtime", "Always"]>,
    MarshallingInfoEnum<CodeGenOpts<"SanitizeAddressUseAfterReturn">, "Runtime">;
defm sanitize_address_poison_custom_array_cookie : BoolOption<"f", "sanitize-address-poison-custom-array-cookie",
  CodeGenOpts<"SanitizeAddressPoisonCustomArrayCookie">, DefaultFalse,
  PosFlag<SetTrue, [], "Enable">, NegFlag<SetFalse, [], "Disable">,
  BothFlags<[], " poisoning array cookies when using custom operator new[] in AddressSanitizer">>,
  Group<f_clang_Group>;
defm sanitize_address_globals_dead_stripping : BoolOption<"f", "sanitize-address-globals-dead-stripping",
  CodeGenOpts<"SanitizeAddressGlobalsDeadStripping">, DefaultFalse,
  PosFlag<SetTrue, [], "Enable linker dead stripping of globals in AddressSanitizer">,
  NegFlag<SetFalse, [], "Disable linker dead stripping of globals in AddressSanitizer">>,
  Group<f_clang_Group>;
defm sanitize_address_use_odr_indicator : BoolOption<"f", "sanitize-address-use-odr-indicator",
  CodeGenOpts<"SanitizeAddressUseOdrIndicator">, DefaultFalse,
  PosFlag<SetTrue, [], "Enable ODR indicator globals to avoid false ODR violation"
            " reports in partially sanitized programs at the cost of an increase in binary size">,
  NegFlag<SetFalse, [], "Disable ODR indicator globals">>,
  Group<f_clang_Group>;
def sanitize_address_destructor_EQ
    : Joined<["-"], "fsanitize-address-destructor=">,
      Flags<[CC1Option]>,
      HelpText<"Set destructor type used in ASan instrumentation">,
      Group<f_clang_Group>,
      Values<"none,global">,
      NormalizedValuesScope<"llvm::AsanDtorKind">,
      NormalizedValues<["None", "Global"]>,
      MarshallingInfoEnum<CodeGenOpts<"SanitizeAddressDtor">, "Global">;
defm sanitize_memory_param_retval
    : BoolFOption<"sanitize-memory-param-retval",
        CodeGenOpts<"SanitizeMemoryParamRetval">,
        DefaultFalse,
        PosFlag<SetTrue, [CC1Option], "Enable">, NegFlag<SetFalse, [], "Disable">,
        BothFlags<[], " detection of uninitialized parameters and return values">>;
//// Note: This flag was introduced when it was necessary to distinguish between
//       ABI for correct codegen.  This is no longer needed, but the flag is
//       not removed since targeting either ABI will behave the same.
//       This way we cause no disturbance to existing scripts & code, and if we
//       want to use this flag in the future we will cause no disturbance then
//       either.
def fsanitize_hwaddress_abi_EQ
    : Joined<["-"], "fsanitize-hwaddress-abi=">,
      Group<f_clang_Group>,
      HelpText<"Select the HWAddressSanitizer ABI to target (interceptor or platform, default interceptor). This option is currently unused.">;
def fsanitize_recover_EQ : CommaJoined<["-"], "fsanitize-recover=">,
                           Group<f_clang_Group>,
                           HelpText<"Enable recovery for specified sanitizers">;
def fno_sanitize_recover_EQ : CommaJoined<["-"], "fno-sanitize-recover=">,
                              Group<f_clang_Group>, Flags<[CoreOption, NoXarchOption]>,
                              HelpText<"Disable recovery for specified sanitizers">;
def fsanitize_recover : Flag<["-"], "fsanitize-recover">, Group<f_clang_Group>,
                        Alias<fsanitize_recover_EQ>, AliasArgs<["all"]>;
def fno_sanitize_recover : Flag<["-"], "fno-sanitize-recover">,
                           Flags<[CoreOption, NoXarchOption]>, Group<f_clang_Group>,
                           Alias<fno_sanitize_recover_EQ>, AliasArgs<["all"]>;
def fsanitize_trap_EQ : CommaJoined<["-"], "fsanitize-trap=">, Group<f_clang_Group>,
                        HelpText<"Enable trapping for specified sanitizers">;
def fno_sanitize_trap_EQ : CommaJoined<["-"], "fno-sanitize-trap=">, Group<f_clang_Group>,
                           Flags<[CoreOption, NoXarchOption]>,
                           HelpText<"Disable trapping for specified sanitizers">;
def fsanitize_trap : Flag<["-"], "fsanitize-trap">, Group<f_clang_Group>,
                     Alias<fsanitize_trap_EQ>, AliasArgs<["all"]>,
                     HelpText<"Enable trapping for all sanitizers">;
def fno_sanitize_trap : Flag<["-"], "fno-sanitize-trap">, Group<f_clang_Group>,
                        Alias<fno_sanitize_trap_EQ>, AliasArgs<["all"]>,
                        Flags<[CoreOption, NoXarchOption]>,
                        HelpText<"Disable trapping for all sanitizers">;
def fsanitize_undefined_trap_on_error
    : Flag<["-"], "fsanitize-undefined-trap-on-error">, Group<f_clang_Group>,
      Alias<fsanitize_trap_EQ>, AliasArgs<["undefined"]>;
def fno_sanitize_undefined_trap_on_error
    : Flag<["-"], "fno-sanitize-undefined-trap-on-error">, Group<f_clang_Group>,
      Alias<fno_sanitize_trap_EQ>, AliasArgs<["undefined"]>;
defm sanitize_minimal_runtime : BoolOption<"f", "sanitize-minimal-runtime",
  CodeGenOpts<"SanitizeMinimalRuntime">, DefaultFalse,
  PosFlag<SetTrue>, NegFlag<SetFalse>>,
  Group<f_clang_Group>;
def fsanitize_link_runtime : Flag<["-"], "fsanitize-link-runtime">,
                           Group<f_clang_Group>;
def fno_sanitize_link_runtime : Flag<["-"], "fno-sanitize-link-runtime">,
                              Group<f_clang_Group>;
def fsanitize_link_cxx_runtime : Flag<["-"], "fsanitize-link-c++-runtime">,
                                 Group<f_clang_Group>;
def fno_sanitize_link_cxx_runtime : Flag<["-"], "fno-sanitize-link-c++-runtime">,
                                    Group<f_clang_Group>;
defm sanitize_cfi_cross_dso : BoolOption<"f", "sanitize-cfi-cross-dso",
  CodeGenOpts<"SanitizeCfiCrossDso">, DefaultFalse,
  PosFlag<SetTrue, [], "Enable">, NegFlag<SetFalse, [CoreOption, NoXarchOption], "Disable">,
  BothFlags<[], " control flow integrity (CFI) checks for cross-DSO calls.">>,
  Group<f_clang_Group>;
def fsanitize_cfi_icall_generalize_pointers : Flag<["-"], "fsanitize-cfi-icall-generalize-pointers">,
                                              Group<f_clang_Group>,
                                              HelpText<"Generalize pointers in CFI indirect call type signature checks">,
                                              MarshallingInfoFlag<CodeGenOpts<"SanitizeCfiICallGeneralizePointers">>;
defm sanitize_cfi_canonical_jump_tables : BoolOption<"f", "sanitize-cfi-canonical-jump-tables",
  CodeGenOpts<"SanitizeCfiCanonicalJumpTables">, DefaultFalse,
  PosFlag<SetTrue, [], "Make">, NegFlag<SetFalse, [CoreOption, NoXarchOption], "Do not make">,
  BothFlags<[], " the jump table addresses canonical in the symbol table">>,
  Group<f_clang_Group>;
defm sanitize_stats : BoolOption<"f", "sanitize-stats",
  CodeGenOpts<"SanitizeStats">, DefaultFalse,
  PosFlag<SetTrue, [], "Enable">, NegFlag<SetFalse, [CoreOption, NoXarchOption], "Disable">,
  BothFlags<[], " sanitizer statistics gathering.">>,
  Group<f_clang_Group>;
def fsanitize_thread_memory_access : Flag<["-"], "fsanitize-thread-memory-access">,
                                     Group<f_clang_Group>,
                                     HelpText<"Enable memory access instrumentation in ThreadSanitizer (default)">;
def fno_sanitize_thread_memory_access : Flag<["-"], "fno-sanitize-thread-memory-access">,
                                        Group<f_clang_Group>,
                                        Flags<[CoreOption, NoXarchOption]>,
                                        HelpText<"Disable memory access instrumentation in ThreadSanitizer">;
def fsanitize_thread_func_entry_exit : Flag<["-"], "fsanitize-thread-func-entry-exit">,
                                       Group<f_clang_Group>,
                                       HelpText<"Enable function entry/exit instrumentation in ThreadSanitizer (default)">;
def fno_sanitize_thread_func_entry_exit : Flag<["-"], "fno-sanitize-thread-func-entry-exit">,
                                          Group<f_clang_Group>,
                                          Flags<[CoreOption, NoXarchOption]>,
                                          HelpText<"Disable function entry/exit instrumentation in ThreadSanitizer">;
def fsanitize_thread_atomics : Flag<["-"], "fsanitize-thread-atomics">,
                               Group<f_clang_Group>,
                               HelpText<"Enable atomic operations instrumentation in ThreadSanitizer (default)">;
def fno_sanitize_thread_atomics : Flag<["-"], "fno-sanitize-thread-atomics">,
                                  Group<f_clang_Group>,
                                  Flags<[CoreOption, NoXarchOption]>,
                                  HelpText<"Disable atomic operations instrumentation in ThreadSanitizer">;
def fsanitize_undefined_strip_path_components_EQ : Joined<["-"], "fsanitize-undefined-strip-path-components=">,
  Group<f_clang_Group>, MetaVarName<"<number>">,
  HelpText<"Strip (or keep only, if negative) a given number of path components "
           "when emitting check metadata.">,
  MarshallingInfoInt<CodeGenOpts<"EmitCheckPathComponentsToStrip">, "0", "int">;

} // end -f[no-]sanitize* flags

def funsafe_math_optimizations : Flag<["-"], "funsafe-math-optimizations">,
  Group<f_Group>;
def fno_unsafe_math_optimizations : Flag<["-"], "fno-unsafe-math-optimizations">,
  Group<f_Group>;
def fassociative_math : Flag<["-"], "fassociative-math">, Group<f_Group>;
def fno_associative_math : Flag<["-"], "fno-associative-math">, Group<f_Group>;
defm reciprocal_math : BoolFOption<"reciprocal-math",
  LangOpts<"AllowRecip">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Allow division operations to be reassociated",
          [menable_unsafe_fp_math.KeyPath]>,
  NegFlag<SetFalse>>;
defm approx_func : BoolFOption<"approx-func", LangOpts<"ApproxFunc">, DefaultFalse,
   PosFlag<SetTrue, [CC1Option], "Allow certain math function calls to be replaced "
           "with an approximately equivalent calculation",
           [menable_unsafe_fp_math.KeyPath]>,
   NegFlag<SetFalse>>;
defm finite_math_only : BoolFOption<"finite-math-only",
  LangOpts<"FiniteMathOnly">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "", [cl_finite_math_only.KeyPath, ffast_math.KeyPath]>,
  NegFlag<SetFalse>>;
defm signed_zeros : BoolFOption<"signed-zeros",
  LangOpts<"NoSignedZero">, DefaultFalse,
  NegFlag<SetTrue, [CC1Option], "Allow optimizations that ignore the sign of floating point zeros",
            [cl_no_signed_zeros.KeyPath, menable_unsafe_fp_math.KeyPath]>,
  PosFlag<SetFalse>>;
def fhonor_nans : Flag<["-"], "fhonor-nans">, Group<f_Group>;
def fno_honor_nans : Flag<["-"], "fno-honor-nans">, Group<f_Group>;
def fhonor_infinities : Flag<["-"], "fhonor-infinities">, Group<f_Group>;
def fno_honor_infinities : Flag<["-"], "fno-honor-infinities">, Group<f_Group>;
// This option was originally misspelt "infinites" [sic].
def : Flag<["-"], "fhonor-infinites">, Alias<fhonor_infinities>;
def : Flag<["-"], "fno-honor-infinites">, Alias<fno_honor_infinities>;
def frounding_math : Flag<["-"], "frounding-math">, Group<f_Group>, Flags<[CC1Option]>,
  MarshallingInfoFlag<LangOpts<"RoundingMath">>,
  Normalizer<"makeFlagToValueNormalizer(llvm::RoundingMode::Dynamic)">;
def fno_rounding_math : Flag<["-"], "fno-rounding-math">, Group<f_Group>, Flags<[CC1Option]>;
def ftrapping_math : Flag<["-"], "ftrapping-math">, Group<f_Group>;
def fno_trapping_math : Flag<["-"], "fno-trapping-math">, Group<f_Group>;
def ffp_contract : Joined<["-"], "ffp-contract=">, Group<f_Group>,
  Flags<[CC1Option]>, HelpText<"Form fused FP ops (e.g. FMAs):"
  " fast (fuses across statements disregarding pragmas)"
  " | on (only fuses in the same statement unless dictated by pragmas)"
  " | off (never fuses)"
  " | fast-honor-pragmas (fuses across statements unless diectated by pragmas)."
  " Default is 'fast' for CUDA, 'fast-honor-pragmas' for HIP, and 'on' otherwise.">,
  Values<"fast,on,off,fast-honor-pragmas">;

defm strict_float_cast_overflow : BoolFOption<"strict-float-cast-overflow",
  CodeGenOpts<"StrictFloatCastOverflow">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Relax language rules and try to match the behavior"
            " of the target's native float-to-int conversion instructions">,
  PosFlag<SetTrue, [], "Assume that overflowing float-to-int casts are undefined (default)">>;

defm protect_parens : BoolFOption<"protect-parens",
  LangOpts<"ProtectParens">, DefaultFalse,
  PosFlag<SetTrue, [CoreOption, CC1Option],
          "Determines whether the optimizer honors parentheses when "
          "floating-point expressions are evaluated">,
  NegFlag<SetFalse>>;

def ffor_scope : Flag<["-"], "ffor-scope">, Group<f_Group>;
def fno_for_scope : Flag<["-"], "fno-for-scope">, Group<f_Group>;

defm rewrite_imports : BoolFOption<"rewrite-imports",
  PreprocessorOutputOpts<"RewriteImports">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option]>, NegFlag<SetFalse>>;
defm rewrite_includes : BoolFOption<"rewrite-includes",
  PreprocessorOutputOpts<"RewriteIncludes">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option]>, NegFlag<SetFalse>>;

defm directives_only : OptInCC1FFlag<"directives-only", "">;

defm delete_null_pointer_checks : BoolFOption<"delete-null-pointer-checks",
  CodeGenOpts<"NullPointerIsValid">, DefaultFalse,
  NegFlag<SetTrue, [CC1Option], "Do not treat usage of null pointers as undefined behavior">,
  PosFlag<SetFalse, [], "Treat usage of null pointers as undefined behavior (default)">,
  BothFlags<[CoreOption]>>;

def frewrite_map_file_EQ : Joined<["-"], "frewrite-map-file=">,
                           Group<f_Group>,
                           Flags<[NoXarchOption, CC1Option]>,
                           MarshallingInfoStringVector<CodeGenOpts<"RewriteMapFiles">>;

defm use_line_directives : BoolFOption<"use-line-directives",
  PreprocessorOutputOpts<"UseLineDirectives">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Use #line in preprocessed output">, NegFlag<SetFalse>>;
defm minimize_whitespace : BoolFOption<"minimize-whitespace",
  PreprocessorOutputOpts<"MinimizeWhitespace">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Minimize whitespace when emitting preprocessor output">, NegFlag<SetFalse>>;

def ffreestanding : Flag<["-"], "ffreestanding">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Assert that the compilation takes place in a freestanding environment">,
  MarshallingInfoFlag<LangOpts<"Freestanding">>;
def fgnuc_version_EQ : Joined<["-"], "fgnuc-version=">, Group<f_Group>,
  HelpText<"Sets various macros to claim compatibility with the given GCC version (default is 4.2.1)">,
  Flags<[CC1Option, CoreOption]>;
// We abuse '-f[no-]gnu-keywords' to force overriding all GNU-extension
// keywords. This behavior is provided by GCC's poorly named '-fasm' flag,
// while a subset (the non-C++ GNU keywords) is provided by GCC's
// '-fgnu-keywords'. Clang conflates the two for simplicity under the single
// name, as it doesn't seem a useful distinction.
defm gnu_keywords : BoolFOption<"gnu-keywords",
  LangOpts<"GNUKeywords">, Default<gnu_mode.KeyPath>,
  PosFlag<SetTrue, [], "Allow GNU-extension keywords regardless of language standard">,
  NegFlag<SetFalse>, BothFlags<[CC1Option]>>;
defm gnu89_inline : BoolFOption<"gnu89-inline",
  LangOpts<"GNUInline">, Default<!strconcat("!", c99.KeyPath, " && !", cplusplus.KeyPath)>,
  PosFlag<SetTrue, [CC1Option], "Use the gnu89 inline semantics">,
  NegFlag<SetFalse>>, ShouldParseIf<!strconcat("!", cplusplus.KeyPath)>;
def fgnu_runtime : Flag<["-"], "fgnu-runtime">, Group<f_Group>,
  HelpText<"Generate output compatible with the standard GNU Objective-C runtime">;
def fheinous_gnu_extensions : Flag<["-"], "fheinous-gnu-extensions">, Flags<[CC1Option]>,
  MarshallingInfoFlag<LangOpts<"HeinousExtensions">>;
def filelist : Separate<["-"], "filelist">, Flags<[LinkerInput]>,
               Group<Link_Group>;
def : Flag<["-"], "findirect-virtual-calls">, Alias<fapple_kext>;
def finline_functions : Flag<["-"], "finline-functions">, Group<f_clang_Group>, Flags<[CC1Option]>,
  HelpText<"Inline suitable functions">;
def finline_hint_functions: Flag<["-"], "finline-hint-functions">, Group<f_clang_Group>, Flags<[CC1Option]>,
  HelpText<"Inline functions which are (explicitly or implicitly) marked inline">;
def finline : Flag<["-"], "finline">, Group<clang_ignored_f_Group>;
defm jmc : BoolFOption<"jmc",
  CodeGenOpts<"JMCInstrument">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable just-my-code debugging">,
  NegFlag<SetFalse>>;
def fglobal_isel : Flag<["-"], "fglobal-isel">, Group<f_clang_Group>,
  HelpText<"Enables the global instruction selector">;
def fexperimental_isel : Flag<["-"], "fexperimental-isel">, Group<f_clang_Group>,
  Alias<fglobal_isel>;
def fno_legacy_pass_manager : Flag<["-"], "fno-legacy-pass-manager">,
  Group<f_clang_Group>, Flags<[CC1Option, NoArgumentUnused]>;
def fexperimental_new_pass_manager : Flag<["-"], "fexperimental-new-pass-manager">,
  Group<f_clang_Group>, Flags<[CC1Option]>, Alias<fno_legacy_pass_manager>;
def fexperimental_strict_floating_point : Flag<["-"], "fexperimental-strict-floating-point">,
  Group<f_clang_Group>, Flags<[CC1Option]>,
  HelpText<"Enables experimental strict floating point in LLVM.">,
  MarshallingInfoFlag<LangOpts<"ExpStrictFP">>;
def finput_charset_EQ : Joined<["-"], "finput-charset=">, Flags<[FlangOption, FC1Option]>, Group<f_Group>,
  HelpText<"Specify the default character set for source files">;
def fexec_charset_EQ : Joined<["-"], "fexec-charset=">, Group<f_Group>;
def finstrument_functions : Flag<["-"], "finstrument-functions">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Generate calls to instrument function entry and exit">,
  MarshallingInfoFlag<CodeGenOpts<"InstrumentFunctions">>;
def finstrument_functions_after_inlining : Flag<["-"], "finstrument-functions-after-inlining">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Like -finstrument-functions, but insert the calls after inlining">,
  MarshallingInfoFlag<CodeGenOpts<"InstrumentFunctionsAfterInlining">>;
def finstrument_function_entry_bare : Flag<["-"], "finstrument-function-entry-bare">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Instrument function entry only, after inlining, without arguments to the instrumentation call">,
  MarshallingInfoFlag<CodeGenOpts<"InstrumentFunctionEntryBare">>;
def fcf_protection_EQ : Joined<["-"], "fcf-protection=">, Flags<[CoreOption, CC1Option]>, Group<f_Group>,
  HelpText<"Instrument control-flow architecture protection">, Values<"return,branch,full,none">;
def fcf_protection : Flag<["-"], "fcf-protection">, Group<f_Group>, Flags<[CoreOption, CC1Option]>,
  Alias<fcf_protection_EQ>, AliasArgs<["full"]>,
  HelpText<"Enable cf-protection in 'full' mode">;
def mibt_seal : Flag<["-"], "mibt-seal">, Group<m_Group>, Flags<[CoreOption, CC1Option]>,
  HelpText<"Optimize fcf-protection=branch/full (requires LTO).">;
def mfunction_return_EQ : Joined<["-"], "mfunction-return=">,
  Group<m_Group>, Flags<[CoreOption, CC1Option]>,
  HelpText<"Replace returns with jumps to ``__x86_return_thunk`` (x86 only, error otherwise)">,
  Values<"keep,thunk-extern">,
  NormalizedValues<["Keep", "Extern"]>,
  NormalizedValuesScope<"llvm::FunctionReturnThunksKind">,
  MarshallingInfoEnum<CodeGenOpts<"FunctionReturnThunks">, "Keep">;

defm xray_instrument : BoolFOption<"xray-instrument",
  LangOpts<"XRayInstrument">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Generate XRay instrumentation sleds on function entry and exit">,
  NegFlag<SetFalse>>;

def fxray_instruction_threshold_EQ :
  JoinedOrSeparate<["-"], "fxray-instruction-threshold=">,
  Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Sets the minimum function size to instrument with XRay">,
  MarshallingInfoInt<CodeGenOpts<"XRayInstructionThreshold">, "200">;
def fxray_instruction_threshold_ :
  JoinedOrSeparate<["-"], "fxray-instruction-threshold">,
  Group<f_Group>, Flags<[CC1Option]>;

def fxray_always_instrument :
  JoinedOrSeparate<["-"], "fxray-always-instrument=">,
  Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"DEPRECATED: Filename defining the whitelist for imbuing the 'always instrument' XRay attribute.">,
  MarshallingInfoStringVector<LangOpts<"XRayAlwaysInstrumentFiles">>;
def fxray_never_instrument :
  JoinedOrSeparate<["-"], "fxray-never-instrument=">,
  Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"DEPRECATED: Filename defining the whitelist for imbuing the 'never instrument' XRay attribute.">,
  MarshallingInfoStringVector<LangOpts<"XRayNeverInstrumentFiles">>;
def fxray_attr_list :
  JoinedOrSeparate<["-"], "fxray-attr-list=">,
  Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Filename defining the list of functions/types for imbuing XRay attributes.">,
  MarshallingInfoStringVector<LangOpts<"XRayAttrListFiles">>;
def fxray_modes :
  JoinedOrSeparate<["-"], "fxray-modes=">,
  Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"List of modes to link in by default into XRay instrumented binaries.">;

defm xray_always_emit_customevents : BoolFOption<"xray-always-emit-customevents",
  LangOpts<"XRayAlwaysEmitCustomEvents">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Always emit __xray_customevent(...) calls"
          " even if the containing function is not always instrumented">,
  NegFlag<SetFalse>>;

defm xray_always_emit_typedevents : BoolFOption<"xray-always-emit-typedevents",
  LangOpts<"XRayAlwaysEmitTypedEvents">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Always emit __xray_typedevent(...) calls"
          " even if the containing function is not always instrumented">,
  NegFlag<SetFalse>>;

defm xray_ignore_loops : BoolFOption<"xray-ignore-loops",
  CodeGenOpts<"XRayIgnoreLoops">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Don't instrument functions with loops"
          " unless they also meet the minimum function size">,
  NegFlag<SetFalse>>;

defm xray_function_index : BoolFOption<"xray-function-index",
  CodeGenOpts<"XRayOmitFunctionIndex">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Omit function index section at the"
          " expense of single-function patching performance">,
  PosFlag<SetTrue>>;

def fxray_link_deps : Flag<["-"], "fxray-link-deps">, Group<f_Group>,
  Flags<[CC1Option]>,
  HelpText<"Tells clang to add the link dependencies for XRay.">;
def fnoxray_link_deps : Flag<["-"], "fnoxray-link-deps">, Group<f_Group>,
  Flags<[CC1Option]>;

def fxray_instrumentation_bundle :
  JoinedOrSeparate<["-"], "fxray-instrumentation-bundle=">,
  Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Select which XRay instrumentation points to emit. Options: all, none, function-entry, function-exit, function, custom. Default is 'all'.  'function' includes both 'function-entry' and 'function-exit'.">;

def fxray_function_groups :
  Joined<["-"], "fxray-function-groups=">,
  Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Only instrument 1 of N groups">,
  MarshallingInfoInt<CodeGenOpts<"XRayTotalFunctionGroups">, "1">;

def fxray_selected_function_group :
  Joined<["-"], "fxray-selected-function-group=">,
  Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"When using -fxray-function-groups, select which group of functions to instrument. Valid range is 0 to fxray-function-groups - 1">,
  MarshallingInfoInt<CodeGenOpts<"XRaySelectedFunctionGroup">, "0">;


defm fine_grained_bitfield_accesses : BoolOption<"f", "fine-grained-bitfield-accesses",
  CodeGenOpts<"FineGrainedBitfieldAccesses">, DefaultFalse,
  PosFlag<SetTrue, [], "Use separate accesses for consecutive bitfield runs with legal widths and alignments.">,
  NegFlag<SetFalse, [], "Use large-integer access for consecutive bitfield runs.">,
  BothFlags<[CC1Option]>>,
  Group<f_clang_Group>;

def fexperimental_relative_cxx_abi_vtables :
  Flag<["-"], "fexperimental-relative-c++-abi-vtables">,
  Group<f_clang_Group>, Flags<[CC1Option]>,
  HelpText<"Use the experimental C++ class ABI for classes with virtual tables">;
def fno_experimental_relative_cxx_abi_vtables :
  Flag<["-"], "fno-experimental-relative-c++-abi-vtables">,
  Group<f_clang_Group>, Flags<[CC1Option]>,
  HelpText<"Do not use the experimental C++ class ABI for classes with virtual tables">;

def fcxx_abi_EQ : Joined<["-"], "fc++-abi=">,
                  Group<f_clang_Group>, Flags<[CC1Option]>,
                  HelpText<"C++ ABI to use. This will override the target C++ ABI.">;

def flat__namespace : Flag<["-"], "flat_namespace">;
def flax_vector_conversions_EQ : Joined<["-"], "flax-vector-conversions=">, Group<f_Group>,
  HelpText<"Enable implicit vector bit-casts">, Values<"none,integer,all">, Flags<[CC1Option]>,
  NormalizedValues<["LangOptions::LaxVectorConversionKind::None",
                    "LangOptions::LaxVectorConversionKind::Integer",
                    "LangOptions::LaxVectorConversionKind::All"]>,
  MarshallingInfoEnum<LangOpts<"LaxVectorConversions">,
                      open_cl.KeyPath #
                          " ? LangOptions::LaxVectorConversionKind::None" #
                          " : LangOptions::LaxVectorConversionKind::All">;
def flax_vector_conversions : Flag<["-"], "flax-vector-conversions">, Group<f_Group>,
  Alias<flax_vector_conversions_EQ>, AliasArgs<["integer"]>;
def flimited_precision_EQ : Joined<["-"], "flimited-precision=">, Group<f_Group>;
def fapple_link_rtlib : Flag<["-"], "fapple-link-rtlib">, Group<f_Group>,
  HelpText<"Force linking the clang builtins runtime library">;
def flto_EQ : Joined<["-"], "flto=">, Flags<[CoreOption, CC1Option]>, Group<f_Group>,
  HelpText<"Set LTO mode">, Values<"thin,full">;
def flto_EQ_jobserver : Flag<["-"], "flto=jobserver">, Group<f_Group>,
  Alias<flto_EQ>, AliasArgs<["full"]>, HelpText<"Enable LTO in 'full' mode">;
def flto_EQ_auto : Flag<["-"], "flto=auto">, Group<f_Group>,
  Alias<flto_EQ>, AliasArgs<["full"]>, HelpText<"Enable LTO in 'full' mode">;
def flto : Flag<["-"], "flto">, Flags<[CoreOption, CC1Option]>, Group<f_Group>,
  Alias<flto_EQ>, AliasArgs<["full"]>, HelpText<"Enable LTO in 'full' mode">;
def fno_lto : Flag<["-"], "fno-lto">, Flags<[CoreOption, CC1Option]>, Group<f_Group>,
  HelpText<"Disable LTO mode (default)">;
def foffload_lto_EQ : Joined<["-"], "foffload-lto=">, Flags<[CoreOption]>, Group<f_Group>,
  HelpText<"Set LTO mode for offload compilation">, Values<"thin,full">;
def foffload_lto : Flag<["-"], "foffload-lto">, Flags<[CoreOption]>, Group<f_Group>,
  Alias<foffload_lto_EQ>, AliasArgs<["full"]>, HelpText<"Enable LTO in 'full' mode for offload compilation">;
def fno_offload_lto : Flag<["-"], "fno-offload-lto">, Flags<[CoreOption]>, Group<f_Group>,
  HelpText<"Disable LTO mode (default) for offload compilation">;
def flto_jobs_EQ : Joined<["-"], "flto-jobs=">,
  Flags<[CC1Option]>, Group<f_Group>,
  HelpText<"Controls the backend parallelism of -flto=thin (default "
           "of 0 means the number of threads will be derived from "
           "the number of CPUs detected)">;
def fthinlto_index_EQ : Joined<["-"], "fthinlto-index=">,
  Flags<[CoreOption, CC1Option]>, Group<f_Group>,
  HelpText<"Perform ThinLTO importing using provided function summary index">;
def fthin_link_bitcode_EQ : Joined<["-"], "fthin-link-bitcode=">,
  Flags<[CoreOption, CC1Option]>, Group<f_Group>,
  HelpText<"Write minimized bitcode to <file> for the ThinLTO thin link only">,
  MarshallingInfoString<CodeGenOpts<"ThinLinkBitcodeFile">>;
def fmacro_backtrace_limit_EQ : Joined<["-"], "fmacro-backtrace-limit=">,
                                Group<f_Group>, Flags<[NoXarchOption, CoreOption]>;
defm merge_all_constants : BoolFOption<"merge-all-constants",
  CodeGenOpts<"MergeAllConstants">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option, CoreOption], "Allow">, NegFlag<SetFalse, [], "Disallow">,
  BothFlags<[], " merging of constants">>;
def fmessage_length_EQ : Joined<["-"], "fmessage-length=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Format message diagnostics so that they fit within N columns">,
  MarshallingInfoInt<DiagnosticOpts<"MessageLength">>;
def frandomize_layout_seed_EQ : Joined<["-"], "frandomize-layout-seed=">,
  MetaVarName<"<seed>">, Group<f_clang_Group>, Flags<[CC1Option]>,
  HelpText<"The seed used by the randomize structure layout feature">;
def frandomize_layout_seed_file_EQ : Joined<["-"], "frandomize-layout-seed-file=">,
  MetaVarName<"<file>">, Group<f_clang_Group>, Flags<[CC1Option]>,
  HelpText<"File holding the seed used by the randomize structure layout feature">;
def fms_compatibility : Flag<["-"], "fms-compatibility">, Group<f_Group>, Flags<[CC1Option, CoreOption]>,
  HelpText<"Enable full Microsoft Visual C++ compatibility">,
  MarshallingInfoFlag<LangOpts<"MSVCCompat">>;
def fms_extensions : Flag<["-"], "fms-extensions">, Group<f_Group>, Flags<[CC1Option, CoreOption]>,
  HelpText<"Accept some non-standard constructs supported by the Microsoft compiler">,
  MarshallingInfoFlag<LangOpts<"MicrosoftExt">>, ImpliedByAnyOf<[fms_compatibility.KeyPath]>;
defm asm_blocks : BoolFOption<"asm-blocks",
  LangOpts<"AsmBlocks">, Default<fms_extensions.KeyPath>,
  PosFlag<SetTrue, [CC1Option]>, NegFlag<SetFalse>>;
def fms_volatile : Flag<["-"], "fms-volatile">, Group<f_Group>, Flags<[CC1Option]>,
  MarshallingInfoFlag<CodeGenOpts<"MSVolatile">>;
def fmsc_version : Joined<["-"], "fmsc-version=">, Group<f_Group>, Flags<[NoXarchOption, CoreOption]>,
  HelpText<"Microsoft compiler version number to report in _MSC_VER (0 = don't define it (default))">;
def fms_compatibility_version
    : Joined<["-"], "fms-compatibility-version=">,
      Group<f_Group>,
      Flags<[ CC1Option, CoreOption ]>,
      HelpText<"Dot-separated value representing the Microsoft compiler "
               "version number to report in _MSC_VER (0 = don't define it "
               "(default))">;
defm delayed_template_parsing : BoolFOption<"delayed-template-parsing",
  LangOpts<"DelayedTemplateParsing">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Parse templated function definitions at the end of the translation unit">,
  NegFlag<SetFalse, [NoXarchOption], "Disable delayed template parsing">,
  BothFlags<[CoreOption]>>;
def fms_memptr_rep_EQ : Joined<["-"], "fms-memptr-rep=">, Group<f_Group>, Flags<[CC1Option]>,
  Values<"single,multiple,virtual">, NormalizedValuesScope<"LangOptions">,
  NormalizedValues<["PPTMK_FullGeneralitySingleInheritance", "PPTMK_FullGeneralityMultipleInheritance",
                    "PPTMK_FullGeneralityVirtualInheritance"]>,
  MarshallingInfoEnum<LangOpts<"MSPointerToMemberRepresentationMethod">, "PPTMK_BestCase">;
def fms_kernel : Flag<["-"], "fms-kernel">, Group<f_Group>, Flags<[CC1Option, NoDriverOption]>,
  MarshallingInfoFlag<LangOpts<"Kernel">>;
// __declspec is enabled by default for the PS4 by the driver, and also
// enabled for Microsoft Extensions or Borland Extensions, here.
//
// FIXME: __declspec is also currently enabled for CUDA, but isn't really a
// CUDA extension. However, it is required for supporting
// __clang_cuda_builtin_vars.h, which uses __declspec(property). Once that has
// been rewritten in terms of something more generic, remove the Opts.CUDA
// term here.
defm declspec : BoolOption<"f", "declspec",
  LangOpts<"DeclSpecKeyword">, DefaultFalse,
  PosFlag<SetTrue, [], "Allow", [fms_extensions.KeyPath, fborland_extensions.KeyPath, cuda.KeyPath]>,
  NegFlag<SetFalse, [], "Disallow">,
  BothFlags<[CC1Option], " __declspec as a keyword">>, Group<f_clang_Group>;
def fmodules_cache_path : Joined<["-"], "fmodules-cache-path=">, Group<i_Group>,
  Flags<[NoXarchOption, CC1Option]>, MetaVarName<"<directory>">,
  HelpText<"Specify the module cache path">;
def fmodules_user_build_path : Separate<["-"], "fmodules-user-build-path">, Group<i_Group>,
  Flags<[NoXarchOption, CC1Option]>, MetaVarName<"<directory>">,
  HelpText<"Specify the module user build path">,
  MarshallingInfoString<HeaderSearchOpts<"ModuleUserBuildPath">>;
def fprebuilt_module_path : Joined<["-"], "fprebuilt-module-path=">, Group<i_Group>,
  Flags<[NoXarchOption, CC1Option]>, MetaVarName<"<directory>">,
  HelpText<"Specify the prebuilt module path">;
defm prebuilt_implicit_modules : BoolFOption<"prebuilt-implicit-modules",
  HeaderSearchOpts<"EnablePrebuiltImplicitModules">, DefaultFalse,
  PosFlag<SetTrue, [], "Look up implicit modules in the prebuilt module path">,
  NegFlag<SetFalse>, BothFlags<[NoXarchOption, CC1Option]>>;

def fmodules_prune_interval : Joined<["-"], "fmodules-prune-interval=">, Group<i_Group>,
  Flags<[CC1Option]>, MetaVarName<"<seconds>">,
  HelpText<"Specify the interval (in seconds) between attempts to prune the module cache">,
  MarshallingInfoInt<HeaderSearchOpts<"ModuleCachePruneInterval">, "7 * 24 * 60 * 60">;
def fmodules_prune_after : Joined<["-"], "fmodules-prune-after=">, Group<i_Group>,
  Flags<[CC1Option]>, MetaVarName<"<seconds>">,
  HelpText<"Specify the interval (in seconds) after which a module file will be considered unused">,
  MarshallingInfoInt<HeaderSearchOpts<"ModuleCachePruneAfter">, "31 * 24 * 60 * 60">;
def fbuild_session_timestamp : Joined<["-"], "fbuild-session-timestamp=">,
  Group<i_Group>, Flags<[CC1Option]>, MetaVarName<"<time since Epoch in seconds>">,
  HelpText<"Time when the current build session started">,
  MarshallingInfoInt<HeaderSearchOpts<"BuildSessionTimestamp">, "0", "uint64_t">;
def fbuild_session_file : Joined<["-"], "fbuild-session-file=">,
  Group<i_Group>, MetaVarName<"<file>">,
  HelpText<"Use the last modification time of <file> as the build session timestamp">;
def fmodules_validate_once_per_build_session : Flag<["-"], "fmodules-validate-once-per-build-session">,
  Group<i_Group>, Flags<[CC1Option]>,
  HelpText<"Don't verify input files for the modules if the module has been "
           "successfully validated or loaded during this build session">,
  MarshallingInfoFlag<HeaderSearchOpts<"ModulesValidateOncePerBuildSession">>;
def fmodules_disable_diagnostic_validation : Flag<["-"], "fmodules-disable-diagnostic-validation">,
  Group<i_Group>, Flags<[CC1Option]>,
  HelpText<"Disable validation of the diagnostic options when loading the module">,
  MarshallingInfoNegativeFlag<HeaderSearchOpts<"ModulesValidateDiagnosticOptions">>;
defm modules_validate_system_headers : BoolOption<"f", "modules-validate-system-headers",
  HeaderSearchOpts<"ModulesValidateSystemHeaders">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Validate the system headers that a module depends on when loading the module">,
  NegFlag<SetFalse, [NoXarchOption]>>, Group<i_Group>;

def fvalidate_ast_input_files_content:
  Flag <["-"], "fvalidate-ast-input-files-content">,
  Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Compute and store the hash of input files used to build an AST."
           " Files with mismatching mtime's are considered valid"
           " if both contents is identical">,
  MarshallingInfoFlag<HeaderSearchOpts<"ValidateASTInputFilesContent">>;
def fmodules_validate_input_files_content:
  Flag <["-"], "fmodules-validate-input-files-content">,
  Group<f_Group>, Flags<[NoXarchOption]>,
  HelpText<"Validate PCM input files based on content if mtime differs">;
def fno_modules_validate_input_files_content:
  Flag <["-"], "fno_modules-validate-input-files-content">,
  Group<f_Group>, Flags<[NoXarchOption]>;
def fpch_validate_input_files_content:
  Flag <["-"], "fpch-validate-input-files-content">,
  Group<f_Group>, Flags<[NoXarchOption]>,
  HelpText<"Validate PCH input files based on content if mtime differs">;
def fno_pch_validate_input_files_content:
  Flag <["-"], "fno_pch-validate-input-files-content">,
  Group<f_Group>, Flags<[NoXarchOption]>;
defm pch_instantiate_templates : BoolFOption<"pch-instantiate-templates",
  LangOpts<"PCHInstantiateTemplates">, DefaultFalse,
  PosFlag<SetTrue, [], "Instantiate templates already while building a PCH">,
  NegFlag<SetFalse>, BothFlags<[CC1Option, CoreOption]>>;
defm pch_codegen: OptInCC1FFlag<"pch-codegen", "Generate ", "Do not generate ",
  "code for uses of this PCH that assumes an explicit object file will be built for the PCH">;
defm pch_debuginfo: OptInCC1FFlag<"pch-debuginfo", "Generate ", "Do not generate ",
  "debug info for types in an object file built from this PCH and do not generate them elsewhere">;

def fimplicit_module_maps : Flag <["-"], "fimplicit-module-maps">, Group<f_Group>,
  Flags<[NoXarchOption, CC1Option, CoreOption]>,
  HelpText<"Implicitly search the file system for module map files.">,
  MarshallingInfoFlag<HeaderSearchOpts<"ImplicitModuleMaps">>;
def fmodules_ts : Flag <["-"], "fmodules-ts">, Group<f_Group>,
  Flags<[CC1Option]>, HelpText<"Enable support for the C++ Modules TS">,
  MarshallingInfoFlag<LangOpts<"ModulesTS">>;
defm modules : BoolFOption<"modules",
  LangOpts<"Modules">, Default<!strconcat(fmodules_ts.KeyPath, "||", fcxx_modules.KeyPath)>,
  PosFlag<SetTrue, [CC1Option], "Enable the 'modules' language feature">,
  NegFlag<SetFalse>, BothFlags<[NoXarchOption, CoreOption]>>;
def fmodule_maps : Flag <["-"], "fmodule-maps">, Flags<[CoreOption]>, Alias<fimplicit_module_maps>;
def fmodule_name_EQ : Joined<["-"], "fmodule-name=">, Group<f_Group>,
  Flags<[NoXarchOption,CC1Option,CoreOption]>, MetaVarName<"<name>">,
  HelpText<"Specify the name of the module to build">,
  MarshallingInfoString<LangOpts<"ModuleName">>;
def fmodule_implementation_of : Separate<["-"], "fmodule-implementation-of">,
  Flags<[CC1Option,CoreOption]>, Alias<fmodule_name_EQ>;
def fsystem_module : Flag<["-"], "fsystem-module">, Flags<[CC1Option,CoreOption]>,
  HelpText<"Build this module as a system module. Only used with -emit-module">,
  MarshallingInfoFlag<FrontendOpts<"IsSystemModule">>;
def fmodule_map_file : Joined<["-"], "fmodule-map-file=">,
  Group<f_Group>, Flags<[NoXarchOption,CC1Option,CoreOption]>, MetaVarName<"<file>">,
  HelpText<"Load this module map file">,
  MarshallingInfoStringVector<FrontendOpts<"ModuleMapFiles">>;
def fmodule_file : Joined<["-"], "fmodule-file=">,
  Group<i_Group>, Flags<[NoXarchOption,CC1Option,CoreOption]>, MetaVarName<"[<name>=]<file>">,
  HelpText<"Specify the mapping of module name to precompiled module file, or load a module file if name is omitted.">;
def fmodules_ignore_macro : Joined<["-"], "fmodules-ignore-macro=">, Group<f_Group>,
  Flags<[CC1Option,CoreOption]>,
  HelpText<"Ignore the definition of the given macro when building and loading modules">;
def fmodules_strict_decluse : Flag <["-"], "fmodules-strict-decluse">, Group<f_Group>,
  Flags<[NoXarchOption,CC1Option,CoreOption]>,
  HelpText<"Like -fmodules-decluse but requires all headers to be in modules">,
  MarshallingInfoFlag<LangOpts<"ModulesStrictDeclUse">>;
defm modules_decluse : BoolFOption<"modules-decluse",
  LangOpts<"ModulesDeclUse">, Default<fmodules_strict_decluse.KeyPath>,
  PosFlag<SetTrue, [CC1Option], "Require declaration of modules used within a module">,
  NegFlag<SetFalse>, BothFlags<[NoXarchOption,CoreOption]>>;
defm modules_search_all : BoolFOption<"modules-search-all",
  LangOpts<"ModulesSearchAll">, DefaultFalse,
  PosFlag<SetTrue, [], "Search even non-imported modules to resolve references">,
  NegFlag<SetFalse>, BothFlags<[NoXarchOption, CC1Option,CoreOption]>>,
  ShouldParseIf<fmodules.KeyPath>;
defm implicit_modules : BoolFOption<"implicit-modules",
  LangOpts<"ImplicitModules">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option]>, PosFlag<SetTrue>, BothFlags<[NoXarchOption,CoreOption]>>;
def fretain_comments_from_system_headers : Flag<["-"], "fretain-comments-from-system-headers">, Group<f_Group>, Flags<[CC1Option]>,
  MarshallingInfoFlag<LangOpts<"RetainCommentsFromSystemHeaders">>;
def fmodule_header : Flag <["-"], "fmodule-header">, Group<f_Group>,
  Flags<[NoXarchOption]>, HelpText<"Build a C++20 Header Unit from a header.">;
def fmodule_header_EQ : Joined<["-"], "fmodule-header=">, Group<f_Group>,
  Flags<[NoXarchOption]>, MetaVarName<"<kind>">,
  HelpText<"Build a C++20 Header Unit from a header that should be found in the user (fmodule-header=user) or system (fmodule-header=system) search path.">;

def fno_knr_functions : Flag<["-"], "fno-knr-functions">, Group<f_Group>,
  MarshallingInfoFlag<LangOpts<"DisableKNRFunctions">>,
  HelpText<"Disable support for K&R C function declarations">,
  Flags<[CC1Option, CoreOption]>;

def fmudflapth : Flag<["-"], "fmudflapth">, Group<f_Group>;
def fmudflap : Flag<["-"], "fmudflap">, Group<f_Group>;
def fnested_functions : Flag<["-"], "fnested-functions">, Group<f_Group>;
def fnext_runtime : Flag<["-"], "fnext-runtime">, Group<f_Group>;
def fno_asm : Flag<["-"], "fno-asm">, Group<f_Group>;
def fno_asynchronous_unwind_tables : Flag<["-"], "fno-asynchronous-unwind-tables">, Group<f_Group>;
def fno_assume_sane_operator_new : Flag<["-"], "fno-assume-sane-operator-new">, Group<f_Group>,
  HelpText<"Don't assume that C++'s global operator new can't alias any pointer">,
  Flags<[CC1Option]>, MarshallingInfoNegativeFlag<CodeGenOpts<"AssumeSaneOperatorNew">>;
def fno_builtin : Flag<["-"], "fno-builtin">, Group<f_Group>, Flags<[CC1Option, CoreOption]>,
  HelpText<"Disable implicit builtin knowledge of functions">;
def fno_builtin_ : Joined<["-"], "fno-builtin-">, Group<f_Group>, Flags<[CC1Option, CoreOption]>,
  HelpText<"Disable implicit builtin knowledge of a specific function">;
def fno_common : Flag<["-"], "fno-common">, Group<f_Group>, Flags<[CC1Option]>,
    HelpText<"Compile common globals like normal definitions">;
defm digraphs : BoolFOption<"digraphs",
  LangOpts<"Digraphs">, Default<std#".hasDigraphs()">,
  PosFlag<SetTrue, [], "Enable alternative token representations '<:', ':>', '<%', '%>', '%:', '%:%:' (default)">,
  NegFlag<SetFalse, [], "Disallow alternative token representations '<:', ':>', '<%', '%>', '%:', '%:%:'">,
  BothFlags<[CC1Option]>>;
def fno_eliminate_unused_debug_symbols : Flag<["-"], "fno-eliminate-unused-debug-symbols">, Group<f_Group>;
def fno_inline_functions : Flag<["-"], "fno-inline-functions">, Group<f_clang_Group>, Flags<[CC1Option]>;
def fno_inline : Flag<["-"], "fno-inline">, Group<f_clang_Group>, Flags<[CC1Option]>;
def fno_global_isel : Flag<["-"], "fno-global-isel">, Group<f_clang_Group>,
  HelpText<"Disables the global instruction selector">;
def fno_experimental_isel : Flag<["-"], "fno-experimental-isel">, Group<f_clang_Group>,
  Alias<fno_global_isel>;
def fveclib : Joined<["-"], "fveclib=">, Group<f_Group>, Flags<[CC1Option]>,
    HelpText<"Use the given vector functions library">,
    Values<"Accelerate,libmvec,MASSV,SVML,Darwin_libsystem_m,none">,
    NormalizedValuesScope<"CodeGenOptions">,
    NormalizedValues<["Accelerate", "LIBMVEC", "MASSV", "SVML",
                      "Darwin_libsystem_m", "NoLibrary"]>,
    MarshallingInfoEnum<CodeGenOpts<"VecLib">, "NoLibrary">;
def fno_lax_vector_conversions : Flag<["-"], "fno-lax-vector-conversions">, Group<f_Group>,
  Alias<flax_vector_conversions_EQ>, AliasArgs<["none"]>;
def fno_implicit_module_maps : Flag <["-"], "fno-implicit-module-maps">, Group<f_Group>,
  Flags<[NoXarchOption]>;
def fno_module_maps : Flag <["-"], "fno-module-maps">, Alias<fno_implicit_module_maps>;
def fno_modules_strict_decluse : Flag <["-"], "fno-strict-modules-decluse">, Group<f_Group>,
  Flags<[NoXarchOption]>;
def fmodule_file_deps : Flag <["-"], "fmodule-file-deps">, Group<f_Group>,
  Flags<[NoXarchOption]>;
def fno_module_file_deps : Flag <["-"], "fno-module-file-deps">, Group<f_Group>,
  Flags<[NoXarchOption]>;
def fno_ms_extensions : Flag<["-"], "fno-ms-extensions">, Group<f_Group>,
  Flags<[CoreOption]>;
def fno_ms_compatibility : Flag<["-"], "fno-ms-compatibility">, Group<f_Group>,
  Flags<[CoreOption]>;
def fno_objc_legacy_dispatch : Flag<["-"], "fno-objc-legacy-dispatch">, Group<f_Group>;
def fno_objc_weak : Flag<["-"], "fno-objc-weak">, Group<f_Group>, Flags<[CC1Option]>;
def fno_omit_frame_pointer : Flag<["-"], "fno-omit-frame-pointer">, Group<f_Group>;
defm operator_names : BoolFOption<"operator-names",
  LangOpts<"CXXOperatorNames">, Default<cplusplus.KeyPath>,
  NegFlag<SetFalse, [CC1Option], "Do not treat C++ operator name keywords as synonyms for operators">,
  PosFlag<SetTrue>>;
def fdiagnostics_absolute_paths : Flag<["-"], "fdiagnostics-absolute-paths">, Group<f_Group>,
  Flags<[CC1Option, CoreOption]>, HelpText<"Print absolute paths in diagnostics">,
  MarshallingInfoFlag<DiagnosticOpts<"AbsolutePath">>;
def fno_stack_protector : Flag<["-"], "fno-stack-protector">, Group<f_Group>,
  HelpText<"Disable the use of stack protectors">;
def fno_strict_aliasing : Flag<["-"], "fno-strict-aliasing">, Group<f_Group>,
  Flags<[NoXarchOption, CoreOption]>;
def fstruct_path_tbaa : Flag<["-"], "fstruct-path-tbaa">, Group<f_Group>;
def fno_struct_path_tbaa : Flag<["-"], "fno-struct-path-tbaa">, Group<f_Group>;
def fno_strict_enums : Flag<["-"], "fno-strict-enums">, Group<f_Group>;
def fno_strict_overflow : Flag<["-"], "fno-strict-overflow">, Group<f_Group>;
def fno_temp_file : Flag<["-"], "fno-temp-file">, Group<f_Group>,
  Flags<[CC1Option, CoreOption]>, HelpText<
  "Directly create compilation output files. This may lead to incorrect incremental builds if the compiler crashes">,
  MarshallingInfoNegativeFlag<FrontendOpts<"UseTemporary">>;
defm use_cxa_atexit : BoolFOption<"use-cxa-atexit",
  CodeGenOpts<"CXAAtExit">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Don't use __cxa_atexit for calling destructors">,
  PosFlag<SetTrue>>;
def fno_unwind_tables : Flag<["-"], "fno-unwind-tables">, Group<f_Group>;
def fno_verbose_asm : Flag<["-"], "fno-verbose-asm">, Group<f_Group>, Flags<[CC1Option]>,
  MarshallingInfoNegativeFlag<CodeGenOpts<"AsmVerbose">>;
def fno_working_directory : Flag<["-"], "fno-working-directory">, Group<f_Group>;
def fno_wrapv : Flag<["-"], "fno-wrapv">, Group<f_Group>;
def fobjc_arc : Flag<["-"], "fobjc-arc">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Synthesize retain and release calls for Objective-C pointers">;
def fno_objc_arc : Flag<["-"], "fno-objc-arc">, Group<f_Group>;
defm objc_encode_cxx_class_template_spec : BoolFOption<"objc-encode-cxx-class-template-spec",
  LangOpts<"EncodeCXXClassTemplateSpec">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Fully encode c++ class template specialization">,
  NegFlag<SetFalse>>;
defm objc_convert_messages_to_runtime_calls : BoolFOption<"objc-convert-messages-to-runtime-calls",
  CodeGenOpts<"ObjCConvertMessagesToRuntimeCalls">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option]>, PosFlag<SetTrue>>;
defm objc_arc_exceptions : BoolFOption<"objc-arc-exceptions",
  CodeGenOpts<"ObjCAutoRefCountExceptions">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Use EH-safe code when synthesizing retains and releases in -fobjc-arc">,
  NegFlag<SetFalse>>;
def fobjc_atdefs : Flag<["-"], "fobjc-atdefs">, Group<clang_ignored_f_Group>;
def fobjc_call_cxx_cdtors : Flag<["-"], "fobjc-call-cxx-cdtors">, Group<clang_ignored_f_Group>;
defm objc_exceptions : BoolFOption<"objc-exceptions",
  LangOpts<"ObjCExceptions">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable Objective-C exceptions">, NegFlag<SetFalse>>;
defm application_extension : BoolFOption<"application-extension",
  LangOpts<"AppExt">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Restrict code to those available for App Extensions">,
  NegFlag<SetFalse>>;
defm relaxed_template_template_args : BoolFOption<"relaxed-template-template-args",
  LangOpts<"RelaxedTemplateTemplateArgs">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable C++17 relaxed template template argument matching">,
  NegFlag<SetFalse>>;
defm sized_deallocation : BoolFOption<"sized-deallocation",
  LangOpts<"SizedDeallocation">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable C++14 sized global deallocation functions">,
  NegFlag<SetFalse>>;
defm aligned_allocation : BoolFOption<"aligned-allocation",
  LangOpts<"AlignedAllocation">, Default<cpp17.KeyPath>,
  PosFlag<SetTrue, [], "Enable C++17 aligned allocation functions">,
  NegFlag<SetFalse>, BothFlags<[CC1Option]>>;
def fnew_alignment_EQ : Joined<["-"], "fnew-alignment=">,
  HelpText<"Specifies the largest alignment guaranteed by '::operator new(size_t)'">,
  MetaVarName<"<align>">, Group<f_Group>, Flags<[CC1Option]>,
  MarshallingInfoInt<LangOpts<"NewAlignOverride">>;
def : Separate<["-"], "fnew-alignment">, Alias<fnew_alignment_EQ>;
def : Flag<["-"], "faligned-new">, Alias<faligned_allocation>;
def : Flag<["-"], "fno-aligned-new">, Alias<fno_aligned_allocation>;
def faligned_new_EQ : Joined<["-"], "faligned-new=">;

def fobjc_legacy_dispatch : Flag<["-"], "fobjc-legacy-dispatch">, Group<f_Group>;
def fobjc_new_property : Flag<["-"], "fobjc-new-property">, Group<clang_ignored_f_Group>;
defm objc_infer_related_result_type : BoolFOption<"objc-infer-related-result-type",
  LangOpts<"ObjCInferRelatedResultType">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "do not infer Objective-C related result type based on method family">,
  PosFlag<SetTrue>>;
def fobjc_link_runtime: Flag<["-"], "fobjc-link-runtime">, Group<f_Group>;
def fobjc_weak : Flag<["-"], "fobjc-weak">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable ARC-style weak references in Objective-C">;

// Objective-C ABI options.
def fobjc_runtime_EQ : Joined<["-"], "fobjc-runtime=">, Group<f_Group>, Flags<[CC1Option, CoreOption]>,
  HelpText<"Specify the target Objective-C runtime kind and version">;
def fobjc_abi_version_EQ : Joined<["-"], "fobjc-abi-version=">, Group<f_Group>;
def fobjc_nonfragile_abi_version_EQ : Joined<["-"], "fobjc-nonfragile-abi-version=">, Group<f_Group>;
def fobjc_nonfragile_abi : Flag<["-"], "fobjc-nonfragile-abi">, Group<f_Group>;
def fno_objc_nonfragile_abi : Flag<["-"], "fno-objc-nonfragile-abi">, Group<f_Group>;

def fobjc_sender_dependent_dispatch : Flag<["-"], "fobjc-sender-dependent-dispatch">, Group<f_Group>;
def fobjc_disable_direct_methods_for_testing :
  Flag<["-"], "fobjc-disable-direct-methods-for-testing">,
  Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Ignore attribute objc_direct so that direct methods can be tested">,
  MarshallingInfoFlag<LangOpts<"ObjCDisableDirectMethodsForTesting">>;
defm objc_avoid_heapify_local_blocks : BoolFOption<"objc-avoid-heapify-local-blocks",
  CodeGenOpts<"ObjCAvoidHeapifyLocalBlocks">, DefaultFalse,
  PosFlag<SetTrue, [], "Try">,
  NegFlag<SetFalse, [], "Don't try">,
  BothFlags<[CC1Option, NoDriverOption], " to avoid heapifying local blocks">>;

def fomit_frame_pointer : Flag<["-"], "fomit-frame-pointer">, Group<f_Group>;
def fopenmp : Flag<["-"], "fopenmp">, Group<f_Group>, Flags<[CC1Option, NoArgumentUnused, FlangOption, FC1Option]>,
  HelpText<"Parse OpenMP pragmas and generate parallel code.">;
def fno_openmp : Flag<["-"], "fno-openmp">, Group<f_Group>, Flags<[NoArgumentUnused]>;
def fopenmp_version_EQ : Joined<["-"], "fopenmp-version=">, Group<f_Group>, Flags<[CC1Option, NoArgumentUnused]>,
  HelpText<"Set OpenMP version (e.g. 45 for OpenMP 4.5, 50 for OpenMP 5.0). Default value is 50.">;
defm openmp_extensions: BoolFOption<"openmp-extensions",
  LangOpts<"OpenMPExtensions">, DefaultTrue,
  PosFlag<SetTrue, [CC1Option, NoArgumentUnused],
          "Enable all Clang extensions for OpenMP directives and clauses">,
  NegFlag<SetFalse, [CC1Option, NoArgumentUnused],
          "Disable all Clang extensions for OpenMP directives and clauses">>;
def fopenmp_EQ : Joined<["-"], "fopenmp=">, Group<f_Group>;
def fopenmp_use_tls : Flag<["-"], "fopenmp-use-tls">, Group<f_Group>,
  Flags<[NoArgumentUnused, HelpHidden]>;
def fnoopenmp_use_tls : Flag<["-"], "fnoopenmp-use-tls">, Group<f_Group>,
  Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fopenmp_targets_EQ : CommaJoined<["-"], "fopenmp-targets=">, Flags<[NoXarchOption, CC1Option]>,
  HelpText<"Specify comma-separated list of triples OpenMP offloading targets to be supported">;
def fopenmp_relocatable_target : Flag<["-"], "fopenmp-relocatable-target">,
  Group<f_Group>, Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fnoopenmp_relocatable_target : Flag<["-"], "fnoopenmp-relocatable-target">,
  Group<f_Group>, Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fopenmp_simd : Flag<["-"], "fopenmp-simd">, Group<f_Group>, Flags<[CC1Option, NoArgumentUnused]>,
  HelpText<"Emit OpenMP code only for SIMD-based constructs.">;
def fopenmp_enable_irbuilder : Flag<["-"], "fopenmp-enable-irbuilder">, Group<f_Group>, Flags<[CC1Option, NoArgumentUnused, HelpHidden]>,
  HelpText<"Use the experimental OpenMP-IR-Builder codegen path.">;
def fno_openmp_simd : Flag<["-"], "fno-openmp-simd">, Group<f_Group>, Flags<[CC1Option, NoArgumentUnused]>;
def fopenmp_cuda_mode : Flag<["-"], "fopenmp-cuda-mode">, Group<f_Group>,
  Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fno_openmp_cuda_mode : Flag<["-"], "fno-openmp-cuda-mode">, Group<f_Group>,
  Flags<[NoArgumentUnused, HelpHidden]>;
def fopenmp_cuda_force_full_runtime : Flag<["-"], "fopenmp-cuda-force-full-runtime">, Group<f_Group>,
  Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fno_openmp_cuda_force_full_runtime : Flag<["-"], "fno-openmp-cuda-force-full-runtime">, Group<f_Group>,
  Flags<[NoArgumentUnused, HelpHidden]>;
def fopenmp_cuda_number_of_sm_EQ : Joined<["-"], "fopenmp-cuda-number-of-sm=">, Group<f_Group>,
  Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fopenmp_cuda_blocks_per_sm_EQ : Joined<["-"], "fopenmp-cuda-blocks-per-sm=">, Group<f_Group>,
  Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fopenmp_cuda_teams_reduction_recs_num_EQ : Joined<["-"], "fopenmp-cuda-teams-reduction-recs-num=">, Group<f_Group>,
  Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fopenmp_target_debug : Flag<["-"], "fopenmp-target-debug">, Group<f_Group>, Flags<[CC1Option, NoArgumentUnused]>,
  HelpText<"Enable debugging in the OpenMP offloading device RTL">;
def fno_openmp_target_debug : Flag<["-"], "fno-openmp-target-debug">, Group<f_Group>, Flags<[NoArgumentUnused]>;
def fopenmp_target_debug_EQ : Joined<["-"], "fopenmp-target-debug=">, Group<f_Group>, Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fopenmp_assume_teams_oversubscription : Flag<["-"], "fopenmp-assume-teams-oversubscription">,
  Group<f_Group>, Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fopenmp_assume_threads_oversubscription : Flag<["-"], "fopenmp-assume-threads-oversubscription">,
  Group<f_Group>, Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fno_openmp_assume_teams_oversubscription : Flag<["-"], "fno-openmp-assume-teams-oversubscription">,
  Group<f_Group>, Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fno_openmp_assume_threads_oversubscription : Flag<["-"], "fno-openmp-assume-threads-oversubscription">,
  Group<f_Group>, Flags<[CC1Option, NoArgumentUnused, HelpHidden]>;
def fopenmp_assume_no_thread_state : Flag<["-"], "fopenmp-assume-no-thread-state">, Group<f_Group>,
  Flags<[CC1Option, NoArgumentUnused, HelpHidden]>,
  HelpText<"Assert no thread in a parallel region modifies an ICV">,
  MarshallingInfoFlag<LangOpts<"OpenMPNoThreadState">>;
def fopenmp_offload_mandatory : Flag<["-"], "fopenmp-offload-mandatory">, Group<f_Group>,
  Flags<[CC1Option, NoArgumentUnused]>,
  HelpText<"Do not create a host fallback if offloading to the device fails.">,
  MarshallingInfoFlag<LangOpts<"OpenMPOffloadMandatory">>;
def fopenmp_target_new_runtime : Flag<["-"], "fopenmp-target-new-runtime">,
  Group<f_Group>, Flags<[CC1Option, HelpHidden]>;
def fno_openmp_target_new_runtime : Flag<["-"], "fno-openmp-target-new-runtime">,
  Group<f_Group>, Flags<[CC1Option, HelpHidden]>;
defm openmp_optimistic_collapse : BoolFOption<"openmp-optimistic-collapse",
  LangOpts<"OpenMPOptimisticCollapse">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option]>, NegFlag<SetFalse>, BothFlags<[NoArgumentUnused, HelpHidden]>>;
def static_openmp: Flag<["-"], "static-openmp">,
  HelpText<"Use the static host OpenMP runtime while linking.">;
def offload_new_driver : Flag<["--"], "offload-new-driver">, Flags<[CC1Option]>, Group<f_Group>,
  MarshallingInfoFlag<LangOpts<"OffloadingNewDriver">>, HelpText<"Use the new driver for offloading compilation.">;
def no_offload_new_driver : Flag<["--"], "no-offload-new-driver">, Flags<[CC1Option]>, Group<f_Group>,
  HelpText<"Don't Use the new driver for offloading compilation.">;
def offload_device_only : Flag<["--"], "offload-device-only">,
  HelpText<"Only compile for the offloading device.">;
def offload_host_only : Flag<["--"], "offload-host-only">,
  HelpText<"Only compile for the offloading host.">;
def offload_host_device : Flag<["--"], "offload-host-device">,
  HelpText<"Only compile for the offloading host.">;
def cuda_device_only : Flag<["--"], "cuda-device-only">, Alias<offload_device_only>,
  HelpText<"Compile CUDA code for device only">;
def cuda_host_only : Flag<["--"], "cuda-host-only">, Alias<offload_host_only>,
  HelpText<"Compile CUDA code for host only. Has no effect on non-CUDA compilations.">;
def cuda_compile_host_device : Flag<["--"], "cuda-compile-host-device">, Alias<offload_host_device>,
  HelpText<"Compile CUDA code for both host and device (default). Has no "
           "effect on non-CUDA compilations.">;
def fopenmp_new_driver : Flag<["-"], "fopenmp-new-driver">, Flags<[CC1Option]>, Group<f_Group>,
  HelpText<"Use the new driver for OpenMP offloading.">;
def fno_openmp_new_driver : Flag<["-"], "fno-openmp-new-driver">, Flags<[CC1Option]>, Group<Action_Group>,
  Alias<no_offload_new_driver>, HelpText<"Don't use the new driver for OpenMP offloading.">;
def fno_optimize_sibling_calls : Flag<["-"], "fno-optimize-sibling-calls">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Disable tail call optimization, keeping the call stack accurate">,
  MarshallingInfoFlag<CodeGenOpts<"DisableTailCalls">>;
def foptimize_sibling_calls : Flag<["-"], "foptimize-sibling-calls">, Group<f_Group>;
defm escaping_block_tail_calls : BoolFOption<"escaping-block-tail-calls",
  CodeGenOpts<"NoEscapingBlockTailCalls">, DefaultFalse,
  NegFlag<SetTrue, [CC1Option]>, PosFlag<SetFalse>>;
def force__cpusubtype__ALL : Flag<["-"], "force_cpusubtype_ALL">;
def force__flat__namespace : Flag<["-"], "force_flat_namespace">;
def force__load : Separate<["-"], "force_load">;
def force_addr : Joined<["-"], "fforce-addr">, Group<clang_ignored_f_Group>;
def foutput_class_dir_EQ : Joined<["-"], "foutput-class-dir=">, Group<f_Group>;
def fpack_struct : Flag<["-"], "fpack-struct">, Group<f_Group>;
def fno_pack_struct : Flag<["-"], "fno-pack-struct">, Group<f_Group>;
def fpack_struct_EQ : Joined<["-"], "fpack-struct=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Specify the default maximum struct packing alignment">,
  MarshallingInfoInt<LangOpts<"PackStruct">>;
def fmax_type_align_EQ : Joined<["-"], "fmax-type-align=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Specify the maximum alignment to enforce on pointers lacking an explicit alignment">,
  MarshallingInfoInt<LangOpts<"MaxTypeAlign">>;
def fno_max_type_align : Flag<["-"], "fno-max-type-align">, Group<f_Group>;
defm pascal_strings : BoolFOption<"pascal-strings",
  LangOpts<"PascalStrings">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Recognize and construct Pascal-style string literals">,
  NegFlag<SetFalse>>;
// Note: This flag has different semantics in the driver and in -cc1. The driver accepts -fpatchable-function-entry=M,N
// and forwards it to -cc1 as -fpatchable-function-entry=M and -fpatchable-function-entry-offset=N. In -cc1, both flags
// are treated as a single integer.
def fpatchable_function_entry_EQ : Joined<["-"], "fpatchable-function-entry=">, Group<f_Group>, Flags<[CC1Option]>,
  MetaVarName<"<N,M>">, HelpText<"Generate M NOPs before function entry and N-M NOPs after function entry">,
  MarshallingInfoInt<CodeGenOpts<"PatchableFunctionEntryCount">>;
def fms_hotpatch : Flag<["-"], "fms-hotpatch">, Group<f_Group>, Flags<[CC1Option, CoreOption]>,
  HelpText<"Ensure that all functions can be hotpatched at runtime">,
  MarshallingInfoFlag<CodeGenOpts<"HotPatch">>;
def fpcc_struct_return : Flag<["-"], "fpcc-struct-return">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Override the default ABI to return all structs on the stack">;
def fpch_preprocess : Flag<["-"], "fpch-preprocess">, Group<f_Group>;
def fpic : Flag<["-"], "fpic">, Group<f_Group>;
def fno_pic : Flag<["-"], "fno-pic">, Group<f_Group>;
def fpie : Flag<["-"], "fpie">, Group<f_Group>;
def fno_pie : Flag<["-"], "fno-pie">, Group<f_Group>;
def fdirect_access_external_data : Flag<["-"], "fdirect-access-external-data">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Don't use GOT indirection to reference external data symbols">;
def fno_direct_access_external_data : Flag<["-"], "fno-direct-access-external-data">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Use GOT indirection to reference external data symbols">;
defm plt : BoolFOption<"plt",
  CodeGenOpts<"NoPLT">, DefaultFalse,
  NegFlag<SetTrue, [CC1Option], "Use GOT indirection instead of PLT to make external function calls (x86 only)">,
  PosFlag<SetFalse>>;
defm ropi : BoolFOption<"ropi",
  LangOpts<"ROPI">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Generate read-only position independent code (ARM only)">,
  NegFlag<SetFalse>>;
defm rwpi : BoolFOption<"rwpi",
  LangOpts<"RWPI">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Generate read-write position independent code (ARM only)">,
  NegFlag<SetFalse>>;
def fplugin_EQ : Joined<["-"], "fplugin=">, Group<f_Group>, Flags<[NoXarchOption]>, MetaVarName<"<dsopath>">,
  HelpText<"Load the named plugin (dynamic shared object)">;
def fplugin_arg : Joined<["-"], "fplugin-arg-">,
  MetaVarName<"<name>-<arg>">,
  HelpText<"Pass <arg> to plugin <name>">;
def fpass_plugin_EQ : Joined<["-"], "fpass-plugin=">,
  Group<f_Group>, Flags<[CC1Option]>, MetaVarName<"<dsopath>">,
  HelpText<"Load pass plugin from a dynamic shared object file (only with new pass manager).">,
  MarshallingInfoStringVector<CodeGenOpts<"PassPlugins">>;
defm preserve_as_comments : BoolFOption<"preserve-as-comments",
  CodeGenOpts<"PreserveAsmComments">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Do not preserve comments in inline assembly">,
  PosFlag<SetTrue>>;
def framework : Separate<["-"], "framework">, Flags<[LinkerInput]>;
def frandom_seed_EQ : Joined<["-"], "frandom-seed=">, Group<clang_ignored_f_Group>;
def freg_struct_return : Flag<["-"], "freg-struct-return">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Override the default ABI to return small structs in registers">;
defm rtti : BoolFOption<"rtti",
  LangOpts<"RTTI">, Default<cplusplus.KeyPath>,
  NegFlag<SetFalse, [CC1Option], "Disable generation of rtti information">,
  PosFlag<SetTrue>>, ShouldParseIf<cplusplus.KeyPath>;
defm rtti_data : BoolFOption<"rtti-data",
  LangOpts<"RTTIData">, Default<frtti.KeyPath>,
  NegFlag<SetFalse, [CC1Option], "Disable generation of RTTI data">,
  PosFlag<SetTrue>>, ShouldParseIf<frtti.KeyPath>;
def : Flag<["-"], "fsched-interblock">, Group<clang_ignored_f_Group>;
defm short_enums : BoolFOption<"short-enums",
  LangOpts<"ShortEnums">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Allocate to an enum type only as many bytes as it"
           " needs for the declared range of possible values">,
  NegFlag<SetFalse>>;
defm char8__t : BoolFOption<"char8_t",
  LangOpts<"Char8">, Default<cpp20.KeyPath>,
  PosFlag<SetTrue, [], "Enable">, NegFlag<SetFalse, [], "Disable">,
  BothFlags<[CC1Option], " C++ builtin type char8_t">>;
def fshort_wchar : Flag<["-"], "fshort-wchar">, Group<f_Group>,
  HelpText<"Force wchar_t to be a short unsigned int">;
def fno_short_wchar : Flag<["-"], "fno-short-wchar">, Group<f_Group>,
  HelpText<"Force wchar_t to be an unsigned int">;
def fshow_overloads_EQ : Joined<["-"], "fshow-overloads=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Which overload candidates to show when overload resolution fails. Defaults to 'all'">,
  Values<"best,all">,
  NormalizedValues<["Ovl_Best", "Ovl_All"]>,
  MarshallingInfoEnum<DiagnosticOpts<"ShowOverloads">, "Ovl_All">;
defm show_column : BoolFOption<"show-column",
  DiagnosticOpts<"ShowColumn">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Do not include column number on diagnostics">,
  PosFlag<SetTrue>>;
defm show_source_location : BoolFOption<"show-source-location",
  DiagnosticOpts<"ShowLocation">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Do not include source location information with diagnostics">,
  PosFlag<SetTrue>>;
defm spell_checking : BoolFOption<"spell-checking",
  LangOpts<"SpellChecking">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Disable spell-checking">, PosFlag<SetTrue>>;
def fspell_checking_limit_EQ : Joined<["-"], "fspell-checking-limit=">, Group<f_Group>;
def fsigned_bitfields : Flag<["-"], "fsigned-bitfields">, Group<f_Group>;
defm signed_char : BoolFOption<"signed-char",
  LangOpts<"CharIsSigned">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "char is unsigned">, PosFlag<SetTrue, [], "char is signed">>,
  ShouldParseIf<!strconcat("!", open_cl.KeyPath)>;
defm split_stack : BoolFOption<"split-stack",
  CodeGenOpts<"EnableSegmentedStacks">, DefaultFalse,
  NegFlag<SetFalse, [], "Wouldn't use segmented stack">,
  PosFlag<SetTrue, [CC1Option], "Use segmented stack">>;
def fstack_protector_all : Flag<["-"], "fstack-protector-all">, Group<f_Group>,
  HelpText<"Enable stack protectors for all functions">;
defm stack_clash_protection : BoolFOption<"stack-clash-protection",
  CodeGenOpts<"StackClashProtector">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable">, NegFlag<SetFalse, [], "Disable">,
  BothFlags<[], " stack clash protection">>;
def fstack_protector_strong : Flag<["-"], "fstack-protector-strong">, Group<f_Group>,
  HelpText<"Enable stack protectors for some functions vulnerable to stack smashing. "
           "Compared to -fstack-protector, this uses a stronger heuristic "
           "that includes functions containing arrays of any size (and any type), "
           "as well as any calls to alloca or the taking of an address from a local variable">;
def fstack_protector : Flag<["-"], "fstack-protector">, Group<f_Group>,
  HelpText<"Enable stack protectors for some functions vulnerable to stack smashing. "
           "This uses a loose heuristic which considers functions vulnerable if they "
           "contain a char (or 8bit integer) array or constant sized calls to alloca "
           ", which are of greater size than ssp-buffer-size (default: 8 bytes). All "
           "variable sized calls to alloca are considered vulnerable. A function with "
           "a stack protector has a guard value added to the stack frame that is "
           "checked on function exit. The guard value must be positioned in the "
           "stack frame such that a buffer overflow from a vulnerable variable will "
           "overwrite the guard value before overwriting the function's return "
           "address. The reference stack guard value is stored in a global variable.">;
def ftrivial_auto_var_init : Joined<["-"], "ftrivial-auto-var-init=">, Group<f_Group>,
  Flags<[CC1Option, CoreOption]>, HelpText<"Initialize trivial automatic stack variables. Defaults to 'uninitialized'">,
  Values<"uninitialized,zero,pattern">,
  NormalizedValuesScope<"LangOptions::TrivialAutoVarInitKind">,
  NormalizedValues<["Uninitialized", "Zero", "Pattern"]>,
  MarshallingInfoEnum<LangOpts<"TrivialAutoVarInit">, "Uninitialized">;
def enable_trivial_var_init_zero : Flag<["-"], "enable-trivial-auto-var-init-zero-knowing-it-will-be-removed-from-clang">,
  Flags<[CC1Option, CoreOption, NoArgumentUnused]>,
  HelpText<"Trivial automatic variable initialization to zero is only here for benchmarks, it'll eventually be removed, and I'm OK with that because I'm only using it to benchmark">;
def ftrivial_auto_var_init_stop_after : Joined<["-"], "ftrivial-auto-var-init-stop-after=">, Group<f_Group>,
  Flags<[CC1Option, CoreOption]>, HelpText<"Stop initializing trivial automatic stack variables after the specified number of instances">,
  MarshallingInfoInt<LangOpts<"TrivialAutoVarInitStopAfter">>;
def fstandalone_debug : Flag<["-"], "fstandalone-debug">, Group<f_Group>, Flags<[CoreOption]>,
  HelpText<"Emit full debug info for all types used by the program">;
def fno_standalone_debug : Flag<["-"], "fno-standalone-debug">, Group<f_Group>, Flags<[CoreOption]>,
  HelpText<"Limit debug information produced to reduce size of debug binary">;
def flimit_debug_info : Flag<["-"], "flimit-debug-info">, Flags<[CoreOption]>, Alias<fno_standalone_debug>;
def fno_limit_debug_info : Flag<["-"], "fno-limit-debug-info">, Flags<[CoreOption]>, Alias<fstandalone_debug>;
def fdebug_macro : Flag<["-"], "fdebug-macro">, Group<f_Group>, Flags<[CoreOption]>,
  HelpText<"Emit macro debug information">;
def fno_debug_macro : Flag<["-"], "fno-debug-macro">, Group<f_Group>, Flags<[CoreOption]>,
  HelpText<"Do not emit macro debug information">;
def fstrict_aliasing : Flag<["-"], "fstrict-aliasing">, Group<f_Group>,
  Flags<[NoXarchOption, CoreOption]>;
def fstrict_enums : Flag<["-"], "fstrict-enums">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable optimizations based on the strict definition of an enum's "
           "value range">,
  MarshallingInfoFlag<CodeGenOpts<"StrictEnums">>;
defm strict_vtable_pointers : BoolFOption<"strict-vtable-pointers",
  CodeGenOpts<"StrictVTablePointers">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable optimizations based on the strict rules for"
            " overwriting polymorphic C++ objects">,
  NegFlag<SetFalse>>;
def fstrict_overflow : Flag<["-"], "fstrict-overflow">, Group<f_Group>;
def fdriver_only : Flag<["-"], "fdriver-only">, Flags<[NoXarchOption, CoreOption]>,
  Group<Action_Group>, HelpText<"Only run the driver.">;
def fsyntax_only : Flag<["-"], "fsyntax-only">,
  Flags<[NoXarchOption,CoreOption,CC1Option,FC1Option]>, Group<Action_Group>;
def ftabstop_EQ : Joined<["-"], "ftabstop=">, Group<f_Group>;
def ftemplate_depth_EQ : Joined<["-"], "ftemplate-depth=">, Group<f_Group>;
def ftemplate_depth_ : Joined<["-"], "ftemplate-depth-">, Group<f_Group>;
def ftemplate_backtrace_limit_EQ : Joined<["-"], "ftemplate-backtrace-limit=">,
                                   Group<f_Group>;
def foperator_arrow_depth_EQ : Joined<["-"], "foperator-arrow-depth=">,
                               Group<f_Group>;

def fsave_optimization_record : Flag<["-"], "fsave-optimization-record">,
  Group<f_Group>, HelpText<"Generate a YAML optimization record file">;
def fsave_optimization_record_EQ : Joined<["-"], "fsave-optimization-record=">,
  Group<f_Group>, HelpText<"Generate an optimization record file in a specific format">,
  MetaVarName<"<format>">;
def fno_save_optimization_record : Flag<["-"], "fno-save-optimization-record">,
  Group<f_Group>, Flags<[NoArgumentUnused]>;
def foptimization_record_file_EQ : Joined<["-"], "foptimization-record-file=">,
  Group<f_Group>,
  HelpText<"Specify the output name of the file containing the optimization remarks. Implies -fsave-optimization-record. On Darwin platforms, this cannot be used with multiple -arch <arch> options.">,
  MetaVarName<"<file>">;
def foptimization_record_passes_EQ : Joined<["-"], "foptimization-record-passes=">,
  Group<f_Group>,
  HelpText<"Only include passes which match a specified regular expression in the generated optimization record (by default, include all passes)">,
  MetaVarName<"<regex>">;

def fvectorize : Flag<["-"], "fvectorize">, Group<f_Group>,
  HelpText<"Enable the loop vectorization passes">;
def fno_vectorize : Flag<["-"], "fno-vectorize">, Group<f_Group>;
def : Flag<["-"], "ftree-vectorize">, Alias<fvectorize>;
def : Flag<["-"], "fno-tree-vectorize">, Alias<fno_vectorize>;
def fslp_vectorize : Flag<["-"], "fslp-vectorize">, Group<f_Group>,
  HelpText<"Enable the superword-level parallelism vectorization passes">;
def fno_slp_vectorize : Flag<["-"], "fno-slp-vectorize">, Group<f_Group>;
def : Flag<["-"], "ftree-slp-vectorize">, Alias<fslp_vectorize>;
def : Flag<["-"], "fno-tree-slp-vectorize">, Alias<fno_slp_vectorize>;
def Wlarge_by_value_copy_def : Flag<["-"], "Wlarge-by-value-copy">,
  HelpText<"Warn if a function definition returns or accepts an object larger "
           "in bytes than a given value">, Flags<[HelpHidden]>;
def Wlarge_by_value_copy_EQ : Joined<["-"], "Wlarge-by-value-copy=">, Flags<[CC1Option]>,
  MarshallingInfoInt<LangOpts<"NumLargeByValueCopy">>;

// These "special" warning flags are effectively processed as f_Group flags by the driver:
// Just silence warnings about -Wlarger-than for now.
def Wlarger_than_EQ : Joined<["-"], "Wlarger-than=">, Group<clang_ignored_f_Group>;
def Wlarger_than_ : Joined<["-"], "Wlarger-than-">, Alias<Wlarger_than_EQ>;

// This is converted to -fwarn-stack-size=N and also passed through by the driver.
// FIXME: The driver should strip out the =<value> when passing W_value_Group through.
def Wframe_larger_than_EQ : Joined<["-"], "Wframe-larger-than=">, Group<W_value_Group>,
                            Flags<[NoXarchOption, CC1Option]>;
def Wframe_larger_than : Flag<["-"], "Wframe-larger-than">, Alias<Wframe_larger_than_EQ>;

def : Flag<["-"], "fterminated-vtables">, Alias<fapple_kext>;
defm threadsafe_statics : BoolFOption<"threadsafe-statics",
  LangOpts<"ThreadsafeStatics">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Do not emit code to make initialization of local statics thread safe">,
  PosFlag<SetTrue>>;
def ftime_report : Flag<["-"], "ftime-report">, Group<f_Group>, Flags<[CC1Option]>,
  MarshallingInfoFlag<CodeGenOpts<"TimePasses">>;
def ftime_report_EQ: Joined<["-"], "ftime-report=">, Group<f_Group>,
  Flags<[CC1Option]>, Values<"per-pass,per-pass-run">,
  MarshallingInfoFlag<CodeGenOpts<"TimePassesPerRun">>,
  HelpText<"(For new pass manager) 'per-pass': one report for each pass; "
           "'per-pass-run': one report for each pass invocation">;
def ftime_trace : Flag<["-"], "ftime-trace">, Group<f_Group>,
  HelpText<"Turn on time profiler. Generates JSON file based on output filename.">,
  DocBrief<[{
Turn on time profiler. Generates JSON file based on output filename. Results
can be analyzed with chrome://tracing or `Speedscope App
<https://www.speedscope.app>`_ for flamegraph visualization.}]>,
  Flags<[CC1Option, CoreOption]>,
  MarshallingInfoFlag<FrontendOpts<"TimeTrace">>;
def ftime_trace_granularity_EQ : Joined<["-"], "ftime-trace-granularity=">, Group<f_Group>,
  HelpText<"Minimum time granularity (in microseconds) traced by time profiler">,
  Flags<[CC1Option, CoreOption]>,
  MarshallingInfoInt<FrontendOpts<"TimeTraceGranularity">, "500u">;
def ftime_trace_EQ : Joined<["-"], "ftime-trace=">, Group<f_Group>,
  HelpText<"Turn on time profiler. Generates JSON file based on output filename. "
           "Specify the path which stores the tracing output file.">,
  DocBrief<[{
  Turn on time profiler. Generates JSON file based on output filename. Results
  can be analyzed with chrome://tracing or `Speedscope App
  <https://www.speedscope.app>`_ for flamegraph visualization.}]>,
  Flags<[CC1Option, CoreOption]>,
  MarshallingInfoString<FrontendOpts<"TimeTracePath">>;
def fproc_stat_report : Joined<["-"], "fproc-stat-report">, Group<f_Group>,
  HelpText<"Print subprocess statistics">;
def fproc_stat_report_EQ : Joined<["-"], "fproc-stat-report=">, Group<f_Group>,
  HelpText<"Save subprocess statistics to the given file">;
def ftlsmodel_EQ : Joined<["-"], "ftls-model=">, Group<f_Group>, Flags<[CC1Option]>,
  Values<"global-dynamic,local-dynamic,initial-exec,local-exec">,
  NormalizedValuesScope<"CodeGenOptions">,
  NormalizedValues<["GeneralDynamicTLSModel", "LocalDynamicTLSModel", "InitialExecTLSModel", "LocalExecTLSModel"]>,
  MarshallingInfoEnum<CodeGenOpts<"DefaultTLSModel">, "GeneralDynamicTLSModel">;
def ftrapv : Flag<["-"], "ftrapv">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Trap on integer overflow">;
def ftrapv_handler_EQ : Joined<["-"], "ftrapv-handler=">, Group<f_Group>,
  MetaVarName<"<function name>">,
  HelpText<"Specify the function to be called on overflow">;
def ftrapv_handler : Separate<["-"], "ftrapv-handler">, Group<f_Group>, Flags<[CC1Option]>;
def ftrap_function_EQ : Joined<["-"], "ftrap-function=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Issue call to specified function rather than a trap instruction">,
  MarshallingInfoString<CodeGenOpts<"TrapFuncName">>;
def funroll_loops : Flag<["-"], "funroll-loops">, Group<f_Group>,
  HelpText<"Turn on loop unroller">, Flags<[CC1Option]>;
def fno_unroll_loops : Flag<["-"], "fno-unroll-loops">, Group<f_Group>,
  HelpText<"Turn off loop unroller">, Flags<[CC1Option]>;
defm reroll_loops : BoolFOption<"reroll-loops",
  CodeGenOpts<"RerollLoops">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Turn on loop reroller">, NegFlag<SetFalse>>;
def ffinite_loops: Flag<["-"],  "ffinite-loops">, Group<f_Group>,
  HelpText<"Assume all loops are finite.">, Flags<[CC1Option]>;
def fno_finite_loops: Flag<["-"], "fno-finite-loops">, Group<f_Group>,
  HelpText<"Do not assume that any loop is finite.">, Flags<[CC1Option]>;

def ftrigraphs : Flag<["-"], "ftrigraphs">, Group<f_Group>,
  HelpText<"Process trigraph sequences">, Flags<[CC1Option]>;
def fno_trigraphs : Flag<["-"], "fno-trigraphs">, Group<f_Group>,
  HelpText<"Do not process trigraph sequences">, Flags<[CC1Option]>;
def funsigned_bitfields : Flag<["-"], "funsigned-bitfields">, Group<f_Group>;
def funsigned_char : Flag<["-"], "funsigned-char">, Group<f_Group>;
def fno_unsigned_char : Flag<["-"], "fno-unsigned-char">;
def funwind_tables : Flag<["-"], "funwind-tables">, Group<f_Group>;
defm register_global_dtors_with_atexit : BoolFOption<"register-global-dtors-with-atexit",
  CodeGenOpts<"RegisterGlobalDtorsWithAtExit">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Use">, NegFlag<SetFalse, [], "Don't use">,
  BothFlags<[], " atexit or __cxa_atexit to register global destructors">>;
defm use_init_array : BoolFOption<"use-init-array",
  CodeGenOpts<"UseInitArray">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Use .ctors/.dtors instead of .init_array/.fini_array">,
  PosFlag<SetTrue>>;
def fno_var_tracking : Flag<["-"], "fno-var-tracking">, Group<clang_ignored_f_Group>;
def fverbose_asm : Flag<["-"], "fverbose-asm">, Group<f_Group>,
  HelpText<"Generate verbose assembly output">;
def dA : Flag<["-"], "dA">, Alias<fverbose_asm>;
defm visibility_from_dllstorageclass : BoolFOption<"visibility-from-dllstorageclass",
  LangOpts<"VisibilityFromDLLStorageClass">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Set the visibility of symbols in the generated code from their DLL storage class">,
  NegFlag<SetFalse>>;
def fvisibility_dllexport_EQ : Joined<["-"], "fvisibility-dllexport=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"The visibility for dllexport definitions [-fvisibility-from-dllstorageclass]">,
  MarshallingInfoVisibility<LangOpts<"DLLExportVisibility">, "DefaultVisibility">,
  ShouldParseIf<fvisibility_from_dllstorageclass.KeyPath>;
def fvisibility_nodllstorageclass_EQ : Joined<["-"], "fvisibility-nodllstorageclass=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"The visibility for definitions without an explicit DLL export class [-fvisibility-from-dllstorageclass]">,
  MarshallingInfoVisibility<LangOpts<"NoDLLStorageClassVisibility">, "HiddenVisibility">,
  ShouldParseIf<fvisibility_from_dllstorageclass.KeyPath>;
def fvisibility_externs_dllimport_EQ : Joined<["-"], "fvisibility-externs-dllimport=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"The visibility for dllimport external declarations [-fvisibility-from-dllstorageclass]">,
  MarshallingInfoVisibility<LangOpts<"ExternDeclDLLImportVisibility">, "DefaultVisibility">,
  ShouldParseIf<fvisibility_from_dllstorageclass.KeyPath>;
def fvisibility_externs_nodllstorageclass_EQ : Joined<["-"], "fvisibility-externs-nodllstorageclass=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"The visibility for external declarations without an explicit DLL dllstorageclass [-fvisibility-from-dllstorageclass]">,
  MarshallingInfoVisibility<LangOpts<"ExternDeclNoDLLStorageClassVisibility">, "HiddenVisibility">,
  ShouldParseIf<fvisibility_from_dllstorageclass.KeyPath>;
def fvisibility_EQ : Joined<["-"], "fvisibility=">, Group<f_Group>,
  HelpText<"Set the default symbol visibility for all global declarations">, Values<"hidden,default">;
defm visibility_inlines_hidden : BoolFOption<"visibility-inlines-hidden",
  LangOpts<"InlineVisibilityHidden">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Give inline C++ member functions hidden visibility by default">,
  NegFlag<SetFalse>>;
defm visibility_inlines_hidden_static_local_var : BoolFOption<"visibility-inlines-hidden-static-local-var",
  LangOpts<"VisibilityInlinesHiddenStaticLocalVar">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "When -fvisibility-inlines-hidden is enabled, static variables in"
            " inline C++ member functions will also be given hidden visibility by default">,
  NegFlag<SetFalse, [], "Disables -fvisibility-inlines-hidden-static-local-var"
         " (this is the default on non-darwin targets)">, BothFlags<[CC1Option]>>;
def fvisibility_ms_compat : Flag<["-"], "fvisibility-ms-compat">, Group<f_Group>,
  HelpText<"Give global types 'default' visibility and global functions and "
           "variables 'hidden' visibility by default">;
def fvisibility_global_new_delete_hidden : Flag<["-"], "fvisibility-global-new-delete-hidden">, Group<f_Group>,
  HelpText<"Give global C++ operator new and delete declarations hidden visibility">, Flags<[CC1Option]>,
  MarshallingInfoFlag<LangOpts<"GlobalAllocationFunctionVisibilityHidden">>;
def mdefault_visibility_export_mapping_EQ : Joined<["-"], "mdefault-visibility-export-mapping=">,
  Values<"none,explicit,all">,
  NormalizedValuesScope<"LangOptions::DefaultVisiblityExportMapping">,
  NormalizedValues<["None", "Explicit", "All"]>,
  HelpText<"Mapping between default visibility and export">,
  Group<m_Group>, Flags<[CC1Option]>,
  MarshallingInfoEnum<LangOpts<"DefaultVisibilityExportMapping">,"None">;
defm new_infallible : BoolFOption<"new-infallible",
  LangOpts<"NewInfallible">, DefaultFalse,
  PosFlag<SetTrue, [], "Enable">, NegFlag<SetFalse, [], "Disable">,
  BothFlags<[CC1Option], " treating throwing global C++ operator new as always returning valid memory "
  "(annotates with __attribute__((returns_nonnull)) and throw()). This is detectable in source.">>;
defm whole_program_vtables : BoolFOption<"whole-program-vtables",
  CodeGenOpts<"WholeProgramVTables">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enables whole-program vtable optimization. Requires -flto">,
  NegFlag<SetFalse>, BothFlags<[CoreOption]>>;
defm split_lto_unit : BoolFOption<"split-lto-unit",
  CodeGenOpts<"EnableSplitLTOUnit">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enables splitting of the LTO unit">,
  NegFlag<SetFalse>, BothFlags<[CoreOption]>>;
defm force_emit_vtables : BoolFOption<"force-emit-vtables",
  CodeGenOpts<"ForceEmitVTables">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Emits more virtual tables to improve devirtualization">,
  NegFlag<SetFalse>, BothFlags<[CoreOption]>>;
defm virtual_function_elimination : BoolFOption<"virtual-function-elimination",
  CodeGenOpts<"VirtualFunctionElimination">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enables dead virtual function elimination optimization. Requires -flto=full">,
  NegFlag<SetFalse>, BothFlags<[CoreOption]>>;

def fwrapv : Flag<["-"], "fwrapv">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Treat signed integer overflow as two's complement">;
def fwritable_strings : Flag<["-"], "fwritable-strings">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Store string literals as writable data">,
  MarshallingInfoFlag<LangOpts<"WritableStrings">>;
defm zero_initialized_in_bss : BoolFOption<"zero-initialized-in-bss",
  CodeGenOpts<"NoZeroInitializedInBSS">, DefaultFalse,
  NegFlag<SetTrue, [CC1Option], "Don't place zero initialized data in BSS">,
  PosFlag<SetFalse>>;
defm function_sections : BoolFOption<"function-sections",
  CodeGenOpts<"FunctionSections">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Place each function in its own section">,
  NegFlag<SetFalse>>;
def fbasic_block_sections_EQ : Joined<["-"], "fbasic-block-sections=">, Group<f_Group>,
  Flags<[CC1Option, CC1AsOption]>,
  HelpText<"Place each function's basic blocks in unique sections (ELF Only)">,
  DocBrief<[{Generate labels for each basic block or place each basic block or a subset of basic blocks in its own section.}]>,
  Values<"all,labels,none,list=">,
  MarshallingInfoString<CodeGenOpts<"BBSections">, [{"none"}]>;
defm data_sections : BoolFOption<"data-sections",
  CodeGenOpts<"DataSections">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Place each data in its own section">, NegFlag<SetFalse>>;
defm stack_size_section : BoolFOption<"stack-size-section",
  CodeGenOpts<"StackSizeSection">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Emit section containing metadata on function stack sizes">,
  NegFlag<SetFalse>>;
def fstack_usage : Flag<["-"], "fstack-usage">, Group<f_Group>,
  HelpText<"Emit .su file containing information on function stack sizes">;
def stack_usage_file : Separate<["-"], "stack-usage-file">,
  Flags<[CC1Option, NoDriverOption]>,
  HelpText<"Filename (or -) to write stack usage output to">,
  MarshallingInfoString<CodeGenOpts<"StackUsageOutput">>;

defm unique_basic_block_section_names : BoolFOption<"unique-basic-block-section-names",
  CodeGenOpts<"UniqueBasicBlockSectionNames">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Use unique names for basic block sections (ELF Only)">,
  NegFlag<SetFalse>>;
defm unique_internal_linkage_names : BoolFOption<"unique-internal-linkage-names",
  CodeGenOpts<"UniqueInternalLinkageNames">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Uniqueify Internal Linkage Symbol Names by appending"
            " the MD5 hash of the module path">,
  NegFlag<SetFalse>>;
defm unique_section_names : BoolFOption<"unique-section-names",
  CodeGenOpts<"UniqueSectionNames">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Don't use unique names for text and data sections">,
  PosFlag<SetTrue>>;

defm split_machine_functions: BoolFOption<"split-machine-functions",
  CodeGenOpts<"SplitMachineFunctions">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable">, NegFlag<SetFalse, [], "Disable">,
  BothFlags<[], " late function splitting using profile information (x86 ELF)">>;

defm strict_return : BoolFOption<"strict-return",
  CodeGenOpts<"StrictReturn">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Don't treat control flow paths that fall off the end"
            " of a non-void function as unreachable">,
  PosFlag<SetTrue>>;

def fenable_matrix : Flag<["-"], "fenable-matrix">, Group<f_Group>,
    Flags<[CC1Option]>,
    HelpText<"Enable matrix data type and related builtin functions">,
    MarshallingInfoFlag<LangOpts<"MatrixTypes">>;

def fzero_call_used_regs_EQ
    : Joined<["-"], "fzero-call-used-regs=">, Group<f_Group>, Flags<[CC1Option]>,
      HelpText<"Clear call-used registers upon function return (AArch64/x86 only)">,
      Values<"skip,used-gpr-arg,used-gpr,used-arg,used,all-gpr-arg,all-gpr,all-arg,all">,
      NormalizedValues<["Skip", "UsedGPRArg", "UsedGPR", "UsedArg", "Used",
                        "AllGPRArg", "AllGPR", "AllArg", "All"]>,
      NormalizedValuesScope<"llvm::ZeroCallUsedRegs::ZeroCallUsedRegsKind">,
      MarshallingInfoEnum<CodeGenOpts<"ZeroCallUsedRegs">, "Skip">;

def fdebug_types_section: Flag <["-"], "fdebug-types-section">, Group<f_Group>,
  HelpText<"Place debug types in their own section (ELF Only)">;
def fno_debug_types_section: Flag<["-"], "fno-debug-types-section">, Group<f_Group>;
defm debug_ranges_base_address : BoolFOption<"debug-ranges-base-address",
  CodeGenOpts<"DebugRangesBaseAddress">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Use DWARF base address selection entries in .debug_ranges">,
  NegFlag<SetFalse>>;
defm split_dwarf_inlining : BoolFOption<"split-dwarf-inlining",
  CodeGenOpts<"SplitDwarfInlining">, DefaultFalse,
  NegFlag<SetFalse, []>,
  PosFlag<SetTrue, [CC1Option], "Provide minimal debug info in the object/executable"
          " to facilitate online symbolication/stack traces in the absence of"
          " .dwo/.dwp files when using Split DWARF">>;
def fdebug_default_version: Joined<["-"], "fdebug-default-version=">, Group<f_Group>,
  HelpText<"Default DWARF version to use, if a -g option caused DWARF debug info to be produced">;
def fdebug_prefix_map_EQ
  : Joined<["-"], "fdebug-prefix-map=">, Group<f_Group>,
    Flags<[CC1Option,CC1AsOption]>,
    HelpText<"remap file source paths in debug info">;
def fcoverage_prefix_map_EQ
  : Joined<["-"], "fcoverage-prefix-map=">, Group<f_Group>,
    Flags<[CC1Option]>,
    HelpText<"remap file source paths in coverage mapping">;
def ffile_prefix_map_EQ
  : Joined<["-"], "ffile-prefix-map=">, Group<f_Group>,
    HelpText<"remap file source paths in debug info, predefined preprocessor "
             "macros and __builtin_FILE(). Implies -ffile-reproducible.">;
def fmacro_prefix_map_EQ
  : Joined<["-"], "fmacro-prefix-map=">, Group<f_Group>, Flags<[CC1Option]>,
    HelpText<"remap file source paths in predefined preprocessor macros and "
             "__builtin_FILE(). Implies -ffile-reproducible.">;
defm force_dwarf_frame : BoolFOption<"force-dwarf-frame",
  CodeGenOpts<"ForceDwarfFrameSection">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Always emit a debug frame section">, NegFlag<SetFalse>>;
def femit_dwarf_unwind_EQ : Joined<["-"], "femit-dwarf-unwind=">,
  Group<f_Group>, Flags<[CC1Option, CC1AsOption]>,
  HelpText<"When to emit DWARF unwind (EH frame) info">,
  Values<"always,no-compact-unwind,default">,
  NormalizedValues<["Always", "NoCompactUnwind", "Default"]>,
  NormalizedValuesScope<"llvm::EmitDwarfUnwindType">,
  MarshallingInfoEnum<CodeGenOpts<"EmitDwarfUnwind">, "Default">;
def g_Flag : Flag<["-"], "g">, Group<g_Group>,
  HelpText<"Generate source-level debug information">;
def gline_tables_only : Flag<["-"], "gline-tables-only">, Group<gN_Group>,
  Flags<[CoreOption]>, HelpText<"Emit debug line number tables only">;
def gline_directives_only : Flag<["-"], "gline-directives-only">, Group<gN_Group>,
  Flags<[CoreOption]>, HelpText<"Emit debug line info directives only">;
def gmlt : Flag<["-"], "gmlt">, Alias<gline_tables_only>;
def g0 : Flag<["-"], "g0">, Group<gN_Group>;
def g1 : Flag<["-"], "g1">, Group<gN_Group>, Alias<gline_tables_only>;
def g2 : Flag<["-"], "g2">, Group<gN_Group>;
def g3 : Flag<["-"], "g3">, Group<gN_Group>;
def ggdb : Flag<["-"], "ggdb">, Group<gTune_Group>;
def ggdb0 : Flag<["-"], "ggdb0">, Group<ggdbN_Group>;
def ggdb1 : Flag<["-"], "ggdb1">, Group<ggdbN_Group>;
def ggdb2 : Flag<["-"], "ggdb2">, Group<ggdbN_Group>;
def ggdb3 : Flag<["-"], "ggdb3">, Group<ggdbN_Group>;
def glldb : Flag<["-"], "glldb">, Group<gTune_Group>;
def gsce : Flag<["-"], "gsce">, Group<gTune_Group>;
def gdbx : Flag<["-"], "gdbx">, Group<gTune_Group>;
// Equivalent to our default dwarf version. Forces usual dwarf emission when
// CodeView is enabled.
def gdwarf : Flag<["-"], "gdwarf">, Group<g_Group>, Flags<[CoreOption]>,
  HelpText<"Generate source-level debug information with the default dwarf version">;
def gdwarf_2 : Flag<["-"], "gdwarf-2">, Group<g_Group>,
  HelpText<"Generate source-level debug information with dwarf version 2">;
def gdwarf_3 : Flag<["-"], "gdwarf-3">, Group<g_Group>,
  HelpText<"Generate source-level debug information with dwarf version 3">;
def gdwarf_4 : Flag<["-"], "gdwarf-4">, Group<g_Group>,
  HelpText<"Generate source-level debug information with dwarf version 4">;
def gdwarf_5 : Flag<["-"], "gdwarf-5">, Group<g_Group>,
  HelpText<"Generate source-level debug information with dwarf version 5">;
def gdwarf64 : Flag<["-"], "gdwarf64">, Group<g_Group>,
  Flags<[CC1Option, CC1AsOption]>,
  HelpText<"Enables DWARF64 format for ELF binaries, if debug information emission is enabled.">,
  MarshallingInfoFlag<CodeGenOpts<"Dwarf64">>;
def gdwarf32 : Flag<["-"], "gdwarf32">, Group<g_Group>,
  Flags<[CC1Option, CC1AsOption]>,
  HelpText<"Enables DWARF32 format for ELF binaries, if debug information emission is enabled.">;

def gcodeview : Flag<["-"], "gcodeview">,
  HelpText<"Generate CodeView debug information">,
  Flags<[CC1Option, CC1AsOption, CoreOption]>,
  MarshallingInfoFlag<CodeGenOpts<"EmitCodeView">>;
defm codeview_ghash : BoolOption<"g", "codeview-ghash",
  CodeGenOpts<"CodeViewGHash">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Emit type record hashes in a .debug$H section">,
  NegFlag<SetFalse>, BothFlags<[CoreOption]>>;
defm inline_line_tables : BoolGOption<"inline-line-tables",
  CodeGenOpts<"NoInlineLineTables">, DefaultFalse,
  NegFlag<SetTrue, [CC1Option], "Don't emit inline line tables.">,
  PosFlag<SetFalse>, BothFlags<[CoreOption]>>;

def gfull : Flag<["-"], "gfull">, Group<g_Group>;
def gused : Flag<["-"], "gused">, Group<g_Group>;
def gstabs : Joined<["-"], "gstabs">, Group<g_Group>, Flags<[Unsupported]>;
def gcoff : Joined<["-"], "gcoff">, Group<g_Group>, Flags<[Unsupported]>;
def gxcoff : Joined<["-"], "gxcoff">, Group<g_Group>, Flags<[Unsupported]>;
def gvms : Joined<["-"], "gvms">, Group<g_Group>, Flags<[Unsupported]>;
def gtoggle : Flag<["-"], "gtoggle">, Group<g_flags_Group>, Flags<[Unsupported]>;
def grecord_command_line : Flag<["-"], "grecord-command-line">,
  Group<g_flags_Group>;
def gno_record_command_line : Flag<["-"], "gno-record-command-line">,
  Group<g_flags_Group>;
def : Flag<["-"], "grecord-gcc-switches">, Alias<grecord_command_line>;
def : Flag<["-"], "gno-record-gcc-switches">, Alias<gno_record_command_line>;
defm strict_dwarf : BoolOption<"g", "strict-dwarf",
  CodeGenOpts<"DebugStrictDwarf">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option]>, NegFlag<SetFalse>, BothFlags<[CoreOption]>>,
  Group<g_flags_Group>;
defm column_info : BoolOption<"g", "column-info",
  CodeGenOpts<"DebugColumnInfo">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option]>, PosFlag<SetTrue>, BothFlags<[CoreOption]>>,
  Group<g_flags_Group>;
def gsplit_dwarf : Flag<["-"], "gsplit-dwarf">, Group<g_flags_Group>;
def gsplit_dwarf_EQ : Joined<["-"], "gsplit-dwarf=">, Group<g_flags_Group>,
  HelpText<"Set DWARF fission mode">,
  Values<"split,single">;
def gno_split_dwarf : Flag<["-"], "gno-split-dwarf">, Group<g_flags_Group>;
def gsimple_template_names : Flag<["-"], "gsimple-template-names">, Group<g_flags_Group>;
def gsimple_template_names_EQ
    : Joined<["-"], "gsimple-template-names=">,
      HelpText<"Use simple template names in DWARF, or include the full "
               "template name with a modified prefix for validation">,
      Values<"simple,mangled">, Flags<[CC1Option, NoDriverOption]>;
def gno_simple_template_names : Flag<["-"], "gno-simple-template-names">,
                                Group<g_flags_Group>;
def ggnu_pubnames : Flag<["-"], "ggnu-pubnames">, Group<g_flags_Group>, Flags<[CC1Option]>;
def gno_gnu_pubnames : Flag<["-"], "gno-gnu-pubnames">, Group<g_flags_Group>;
def gpubnames : Flag<["-"], "gpubnames">, Group<g_flags_Group>, Flags<[CC1Option]>;
def gno_pubnames : Flag<["-"], "gno-pubnames">, Group<g_flags_Group>;
def gdwarf_aranges : Flag<["-"], "gdwarf-aranges">, Group<g_flags_Group>;
def gmodules : Flag <["-"], "gmodules">, Group<gN_Group>,
  HelpText<"Generate debug info with external references to clang modules"
           " or precompiled headers">;
def gz_EQ : Joined<["-"], "gz=">, Group<g_flags_Group>,
    HelpText<"DWARF debug sections compression type">;
def gz : Flag<["-"], "gz">, Alias<gz_EQ>, AliasArgs<["zlib"]>, Group<g_flags_Group>;
def gembed_source : Flag<["-"], "gembed-source">, Group<g_flags_Group>, Flags<[CC1Option]>,
    HelpText<"Embed source text in DWARF debug sections">,
    MarshallingInfoFlag<CodeGenOpts<"EmbedSource">>;
def gno_embed_source : Flag<["-"], "gno-embed-source">, Group<g_flags_Group>,
    Flags<[NoXarchOption]>,
    HelpText<"Restore the default behavior of not embedding source text in DWARF debug sections">;
def headerpad__max__install__names : Joined<["-"], "headerpad_max_install_names">;
def help : Flag<["-", "--"], "help">, Flags<[CC1Option,CC1AsOption, FC1Option,
    FlangOption]>, HelpText<"Display available options">,
    MarshallingInfoFlag<FrontendOpts<"ShowHelp">>;
def ibuiltininc : Flag<["-"], "ibuiltininc">,
  HelpText<"Enable builtin #include directories even when -nostdinc is used "
           "before or after -ibuiltininc. "
           "Using -nobuiltininc after the option disables it">;
def index_header_map : Flag<["-"], "index-header-map">, Flags<[CC1Option]>,
  HelpText<"Make the next included directory (-I or -F) an indexer header map">;
def idirafter : JoinedOrSeparate<["-"], "idirafter">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Add directory to AFTER include search path">;
def iframework : JoinedOrSeparate<["-"], "iframework">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Add directory to SYSTEM framework search path">;
def iframeworkwithsysroot : JoinedOrSeparate<["-"], "iframeworkwithsysroot">,
  Group<clang_i_Group>,
  HelpText<"Add directory to SYSTEM framework search path, "
           "absolute paths are relative to -isysroot">,
  MetaVarName<"<directory>">, Flags<[CC1Option]>;
def imacros : JoinedOrSeparate<["-", "--"], "imacros">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Include macros from file before parsing">, MetaVarName<"<file>">,
  MarshallingInfoStringVector<PreprocessorOpts<"MacroIncludes">>;
def image__base : Separate<["-"], "image_base">;
def include_ : JoinedOrSeparate<["-", "--"], "include">, Group<clang_i_Group>, EnumName<"include">,
    MetaVarName<"<file>">, HelpText<"Include file before parsing">, Flags<[CC1Option]>;
def include_pch : Separate<["-"], "include-pch">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Include precompiled header file">, MetaVarName<"<file>">,
  MarshallingInfoString<PreprocessorOpts<"ImplicitPCHInclude">>;
def relocatable_pch : Flag<["-", "--"], "relocatable-pch">, Flags<[CC1Option]>,
  HelpText<"Whether to build a relocatable precompiled header">,
  MarshallingInfoFlag<FrontendOpts<"RelocatablePCH">>;
def verify_pch : Flag<["-"], "verify-pch">, Group<Action_Group>, Flags<[CC1Option]>,
  HelpText<"Load and verify that a pre-compiled header file is not stale">;
def init : Separate<["-"], "init">;
def install__name : Separate<["-"], "install_name">;
def iprefix : JoinedOrSeparate<["-"], "iprefix">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Set the -iwithprefix/-iwithprefixbefore prefix">, MetaVarName<"<dir>">;
def iquote : JoinedOrSeparate<["-"], "iquote">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Add directory to QUOTE include search path">, MetaVarName<"<directory>">;
def isysroot : JoinedOrSeparate<["-"], "isysroot">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Set the system root directory (usually /)">, MetaVarName<"<dir>">,
  MarshallingInfoString<HeaderSearchOpts<"Sysroot">, [{"/"}]>;
def isystem : JoinedOrSeparate<["-"], "isystem">, Group<clang_i_Group>,
  Flags<[CC1Option]>,
  HelpText<"Add directory to SYSTEM include search path">, MetaVarName<"<directory>">;
def isystem_after : JoinedOrSeparate<["-"], "isystem-after">,
  Group<clang_i_Group>, Flags<[NoXarchOption]>, MetaVarName<"<directory>">,
  HelpText<"Add directory to end of the SYSTEM include search path">;
def iwithprefixbefore : JoinedOrSeparate<["-"], "iwithprefixbefore">, Group<clang_i_Group>,
  HelpText<"Set directory to include search path with prefix">, MetaVarName<"<dir>">,
  Flags<[CC1Option]>;
def iwithprefix : JoinedOrSeparate<["-"], "iwithprefix">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Set directory to SYSTEM include search path with prefix">, MetaVarName<"<dir>">;
def iwithsysroot : JoinedOrSeparate<["-"], "iwithsysroot">, Group<clang_i_Group>,
  HelpText<"Add directory to SYSTEM include search path, "
           "absolute paths are relative to -isysroot">, MetaVarName<"<directory>">,
  Flags<[CC1Option]>;
def ivfsoverlay : JoinedOrSeparate<["-"], "ivfsoverlay">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Overlay the virtual filesystem described by file over the real file system">;
def imultilib : Separate<["-"], "imultilib">, Group<gfortran_Group>;
def keep__private__externs : Flag<["-"], "keep_private_externs">;
def l : JoinedOrSeparate<["-"], "l">, Flags<[LinkerInput, RenderJoined]>,
        Group<Link_Group>;
def lazy__framework : Separate<["-"], "lazy_framework">, Flags<[LinkerInput]>;
def lazy__library : Separate<["-"], "lazy_library">, Flags<[LinkerInput]>;
def mlittle_endian : Flag<["-"], "mlittle-endian">, Flags<[NoXarchOption]>;
def EL : Flag<["-"], "EL">, Alias<mlittle_endian>;
def mbig_endian : Flag<["-"], "mbig-endian">, Flags<[NoXarchOption]>;
def EB : Flag<["-"], "EB">, Alias<mbig_endian>;
def m16 : Flag<["-"], "m16">, Group<m_Group>, Flags<[NoXarchOption, CoreOption]>;
def m32 : Flag<["-"], "m32">, Group<m_Group>, Flags<[NoXarchOption, CoreOption]>;
def mqdsp6_compat : Flag<["-"], "mqdsp6-compat">, Group<m_Group>, Flags<[NoXarchOption,CC1Option]>,
  HelpText<"Enable hexagon-qdsp6 backward compatibility">,
  MarshallingInfoFlag<LangOpts<"HexagonQdsp6Compat">>;
def m64 : Flag<["-"], "m64">, Group<m_Group>, Flags<[NoXarchOption, CoreOption]>;
def mx32 : Flag<["-"], "mx32">, Group<m_Group>, Flags<[NoXarchOption, CoreOption]>;
def mabi_EQ : Joined<["-"], "mabi=">, Group<m_Group>;
def miamcu : Flag<["-"], "miamcu">, Group<m_Group>, Flags<[NoXarchOption, CoreOption]>,
  HelpText<"Use Intel MCU ABI">;
def mno_iamcu : Flag<["-"], "mno-iamcu">, Group<m_Group>, Flags<[NoXarchOption, CoreOption]>;
def malign_functions_EQ : Joined<["-"], "malign-functions=">, Group<clang_ignored_m_Group>;
def malign_loops_EQ : Joined<["-"], "malign-loops=">, Group<clang_ignored_m_Group>;
def malign_jumps_EQ : Joined<["-"], "malign-jumps=">, Group<clang_ignored_m_Group>;
def malign_branch_EQ : CommaJoined<["-"], "malign-branch=">, Group<m_Group>, Flags<[NoXarchOption]>,
  HelpText<"Specify types of branches to align">;
def malign_branch_boundary_EQ : Joined<["-"], "malign-branch-boundary=">, Group<m_Group>, Flags<[NoXarchOption]>,
  HelpText<"Specify the boundary's size to align branches">;
def mpad_max_prefix_size_EQ : Joined<["-"], "mpad-max-prefix-size=">, Group<m_Group>, Flags<[NoXarchOption]>,
  HelpText<"Specify maximum number of prefixes to use for padding">;
def mbranches_within_32B_boundaries : Flag<["-"], "mbranches-within-32B-boundaries">, Flags<[NoXarchOption]>, Group<m_Group>,
  HelpText<"Align selected branches (fused, jcc, jmp) within 32-byte boundary">;
def mfancy_math_387 : Flag<["-"], "mfancy-math-387">, Group<clang_ignored_m_Group>;
def mlong_calls : Flag<["-"], "mlong-calls">, Group<m_Group>,
  HelpText<"Generate branches with extended addressability, usually via indirect jumps.">;
def mdouble_EQ : Joined<["-"], "mdouble=">, Group<m_Group>,
  MetaVarName<"<n">, Values<"32,64">, Flags<[CC1Option]>,
  HelpText<"Force double to be <n> bits">,
  MarshallingInfoInt<LangOpts<"DoubleSize">, "0">;
def LongDouble_Group : OptionGroup<"<LongDouble group>">, Group<m_Group>,
  DocName<"Long double flags">,
  DocBrief<[{Selects the long double implementation}]>;
def mlong_double_64 : Flag<["-"], "mlong-double-64">, Group<LongDouble_Group>, Flags<[CC1Option]>,
  HelpText<"Force long double to be 64 bits">;
def mlong_double_80 : Flag<["-"], "mlong-double-80">, Group<LongDouble_Group>, Flags<[CC1Option]>,
  HelpText<"Force long double to be 80 bits, padded to 128 bits for storage">;
def mlong_double_128 : Flag<["-"], "mlong-double-128">, Group<LongDouble_Group>, Flags<[CC1Option]>,
  HelpText<"Force long double to be 128 bits">;
def mno_long_calls : Flag<["-"], "mno-long-calls">, Group<m_Group>,
  HelpText<"Restore the default behaviour of not generating long calls">;
def mexecute_only : Flag<["-"], "mexecute-only">, Group<m_arm_Features_Group>,
  HelpText<"Disallow generation of data access to code sections (ARM only)">;
def mno_execute_only : Flag<["-"], "mno-execute-only">, Group<m_arm_Features_Group>,
  HelpText<"Allow generation of data access to code sections (ARM only)">;
def mtp_mode_EQ : Joined<["-"], "mtp=">, Group<m_arm_Features_Group>, Values<"soft,cp15,el0,el1,el2,el3">,
  HelpText<"Thread pointer access method (AArch32/AArch64 only)">;
def mpure_code : Flag<["-"], "mpure-code">, Alias<mexecute_only>; // Alias for GCC compatibility
def mno_pure_code : Flag<["-"], "mno-pure-code">, Alias<mno_execute_only>;
def mtvos_version_min_EQ : Joined<["-"], "mtvos-version-min=">, Group<m_Group>;
def mappletvos_version_min_EQ : Joined<["-"], "mappletvos-version-min=">, Alias<mtvos_version_min_EQ>;
def mtvos_simulator_version_min_EQ : Joined<["-"], "mtvos-simulator-version-min=">;
def mappletvsimulator_version_min_EQ : Joined<["-"], "mappletvsimulator-version-min=">, Alias<mtvos_simulator_version_min_EQ>;
def mwatchos_version_min_EQ : Joined<["-"], "mwatchos-version-min=">, Group<m_Group>;
def mwatchos_simulator_version_min_EQ : Joined<["-"], "mwatchos-simulator-version-min=">;
def mwatchsimulator_version_min_EQ : Joined<["-"], "mwatchsimulator-version-min=">, Alias<mwatchos_simulator_version_min_EQ>;
def march_EQ : Joined<["-"], "march=">, Group<m_Group>, Flags<[CoreOption]>;
def masm_EQ : Joined<["-"], "masm=">, Group<m_Group>, Flags<[NoXarchOption]>;
def inline_asm_EQ : Joined<["-"], "inline-asm=">, Group<m_Group>, Flags<[CC1Option]>,
  Values<"att,intel">,
  NormalizedValuesScope<"CodeGenOptions">, NormalizedValues<["IAD_ATT", "IAD_Intel"]>,
  MarshallingInfoEnum<CodeGenOpts<"InlineAsmDialect">, "IAD_ATT">;
def mcmodel_EQ : Joined<["-"], "mcmodel=">, Group<m_Group>, Flags<[CC1Option]>,
  MarshallingInfoString<TargetOpts<"CodeModel">, [{"default"}]>;
def mtls_size_EQ : Joined<["-"], "mtls-size=">, Group<m_Group>, Flags<[NoXarchOption, CC1Option]>,
  HelpText<"Specify bit size of immediate TLS offsets (AArch64 ELF only): "
           "12 (for 4KB) | 24 (for 16MB, default) | 32 (for 4GB) | 48 (for 256TB, needs -mcmodel=large)">,
  MarshallingInfoInt<CodeGenOpts<"TLSSize">>;
def mimplicit_it_EQ : Joined<["-"], "mimplicit-it=">, Group<m_Group>;
def mdefault_build_attributes : Joined<["-"], "mdefault-build-attributes">, Group<m_Group>;
def mno_default_build_attributes : Joined<["-"], "mno-default-build-attributes">, Group<m_Group>;
def mconstant_cfstrings : Flag<["-"], "mconstant-cfstrings">, Group<clang_ignored_m_Group>;
def mconsole : Joined<["-"], "mconsole">, Group<m_Group>, Flags<[NoXarchOption]>;
def mwindows : Joined<["-"], "mwindows">, Group<m_Group>, Flags<[NoXarchOption]>;
def mdll : Joined<["-"], "mdll">, Group<m_Group>, Flags<[NoXarchOption]>;
def municode : Joined<["-"], "municode">, Group<m_Group>, Flags<[NoXarchOption]>;
def mthreads : Joined<["-"], "mthreads">, Group<m_Group>, Flags<[NoXarchOption]>;
def mcpu_EQ : Joined<["-"], "mcpu=">, Group<m_Group>;
def mmcu_EQ : Joined<["-"], "mmcu=">, Group<m_Group>;
def msim : Flag<["-"], "msim">, Group<m_Group>;
def mdynamic_no_pic : Joined<["-"], "mdynamic-no-pic">, Group<m_Group>;
def mfix_and_continue : Flag<["-"], "mfix-and-continue">, Group<clang_ignored_m_Group>;
def mieee_fp : Flag<["-"], "mieee-fp">, Group<clang_ignored_m_Group>;
def minline_all_stringops : Flag<["-"], "minline-all-stringops">, Group<clang_ignored_m_Group>;
def mno_inline_all_stringops : Flag<["-"], "mno-inline-all-stringops">, Group<clang_ignored_m_Group>;
def malign_double : Flag<["-"], "malign-double">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Align doubles to two words in structs (x86 only)">,
  MarshallingInfoFlag<LangOpts<"AlignDouble">>;
def mfloat_abi_EQ : Joined<["-"], "mfloat-abi=">, Group<m_Group>, Values<"soft,softfp,hard">;
def mfpmath_EQ : Joined<["-"], "mfpmath=">, Group<m_Group>;
def mfpu_EQ : Joined<["-"], "mfpu=">, Group<m_Group>;
def mhwdiv_EQ : Joined<["-"], "mhwdiv=">, Group<m_Group>;
def mhwmult_EQ : Joined<["-"], "mhwmult=">, Group<m_Group>;
def mglobal_merge : Flag<["-"], "mglobal-merge">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Enable merging of globals">;
def mhard_float : Flag<["-"], "mhard-float">, Group<m_Group>;
def mios_version_min_EQ : Joined<["-"], "mios-version-min=">,
  Group<m_Group>, HelpText<"Set iOS deployment target">;
def : Joined<["-"], "miphoneos-version-min=">,
  Group<m_Group>, Alias<mios_version_min_EQ>;
def mios_simulator_version_min_EQ : Joined<["-"], "mios-simulator-version-min=">;
def : Joined<["-"], "miphonesimulator-version-min=">, Alias<mios_simulator_version_min_EQ>;
def mkernel : Flag<["-"], "mkernel">, Group<m_Group>;
def mlinker_version_EQ : Joined<["-"], "mlinker-version=">,
  Flags<[NoXarchOption]>;
def mllvm : Separate<["-"], "mllvm">,Flags<[CC1Option,CC1AsOption,CoreOption,FC1Option,FlangOption]>,
  HelpText<"Additional arguments to forward to LLVM's option processing">,
  MarshallingInfoStringVector<FrontendOpts<"LLVMArgs">>;
def mmlir : Separate<["-"], "mmlir">, Flags<[CoreOption,FC1Option,FlangOption]>,
  HelpText<"Additional arguments to forward to MLIR's option processing">;
def ffuchsia_api_level_EQ : Joined<["-"], "ffuchsia-api-level=">,
  Group<m_Group>, Flags<[CC1Option]>, HelpText<"Set Fuchsia API level">,
  MarshallingInfoInt<LangOpts<"FuchsiaAPILevel">>;
def mmacos_version_min_EQ : Joined<["-"], "mmacos-version-min=">,
  Group<m_Group>, HelpText<"Set macOS deployment target">;
def : Joined<["-"], "mmacosx-version-min=">,
  Group<m_Group>, Alias<mmacos_version_min_EQ>;
def mms_bitfields : Flag<["-"], "mms-bitfields">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Set the default structure layout to be compatible with the Microsoft compiler standard">,
  MarshallingInfoFlag<LangOpts<"MSBitfields">>;
def moutline : Flag<["-"], "moutline">, Group<f_clang_Group>, Flags<[CC1Option]>,
    HelpText<"Enable function outlining (AArch64 only)">;
def mno_outline : Flag<["-"], "mno-outline">, Group<f_clang_Group>, Flags<[CC1Option]>,
    HelpText<"Disable function outlining (AArch64 only)">;
def mno_ms_bitfields : Flag<["-"], "mno-ms-bitfields">, Group<m_Group>,
  HelpText<"Do not set the default structure layout to be compatible with the Microsoft compiler standard">;
def mskip_rax_setup : Flag<["-"], "mskip-rax-setup">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Skip setting up RAX register when passing variable arguments (x86 only)">,
  MarshallingInfoFlag<CodeGenOpts<"SkipRaxSetup">>;
def mno_skip_rax_setup : Flag<["-"], "mno-skip-rax-setup">, Group<m_Group>, Flags<[CC1Option]>;
def mstackrealign : Flag<["-"], "mstackrealign">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Force realign the stack at entry to every function">,
  MarshallingInfoFlag<CodeGenOpts<"StackRealignment">>;
def mstack_alignment : Joined<["-"], "mstack-alignment=">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Set the stack alignment">,
  MarshallingInfoInt<CodeGenOpts<"StackAlignment">>;
def mstack_probe_size : Joined<["-"], "mstack-probe-size=">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Set the stack probe size">,
  MarshallingInfoInt<CodeGenOpts<"StackProbeSize">, "4096">;
def mstack_arg_probe : Flag<["-"], "mstack-arg-probe">, Group<m_Group>,
  HelpText<"Enable stack probes">;
def mno_stack_arg_probe : Flag<["-"], "mno-stack-arg-probe">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Disable stack probes which are enabled by default">,
  MarshallingInfoFlag<CodeGenOpts<"NoStackArgProbe">>;
def mthread_model : Separate<["-"], "mthread-model">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"The thread model to use. Defaults to 'posix')">, Values<"posix,single">,
  NormalizedValues<["POSIX", "Single"]>, NormalizedValuesScope<"LangOptions::ThreadModelKind">,
  MarshallingInfoEnum<LangOpts<"ThreadModel">, "POSIX">;
def meabi : Separate<["-"], "meabi">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Set EABI type. Default depends on triple)">, Values<"default,4,5,gnu">,
  MarshallingInfoEnum<TargetOpts<"EABIVersion">, "Default">,
  NormalizedValuesScope<"llvm::EABI">,
  NormalizedValues<["Default", "EABI4", "EABI5", "GNU"]>;
def mtargetos_EQ : Joined<["-"], "mtargetos=">, Group<m_Group>,
  HelpText<"Set the deployment target to be the specified OS and OS version">;

def mno_constant_cfstrings : Flag<["-"], "mno-constant-cfstrings">, Group<m_Group>;
def mno_global_merge : Flag<["-"], "mno-global-merge">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Disable merging of globals">;
def mno_pascal_strings : Flag<["-"], "mno-pascal-strings">,
  Alias<fno_pascal_strings>;
def mno_red_zone : Flag<["-"], "mno-red-zone">, Group<m_Group>;
def mno_tls_direct_seg_refs : Flag<["-"], "mno-tls-direct-seg-refs">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Disable direct TLS access through segment registers">,
  MarshallingInfoFlag<CodeGenOpts<"IndirectTlsSegRefs">>;
def mno_relax_all : Flag<["-"], "mno-relax-all">, Group<m_Group>;
def mno_rtd: Flag<["-"], "mno-rtd">, Group<m_Group>;
def mno_soft_float : Flag<["-"], "mno-soft-float">, Group<m_Group>;
def mno_stackrealign : Flag<["-"], "mno-stackrealign">, Group<m_Group>;

def mretpoline : Flag<["-"], "mretpoline">, Group<m_Group>, Flags<[CoreOption,NoXarchOption]>;
def mno_retpoline : Flag<["-"], "mno-retpoline">, Group<m_Group>, Flags<[CoreOption,NoXarchOption]>;
defm speculative_load_hardening : BoolOption<"m", "speculative-load-hardening",
  CodeGenOpts<"SpeculativeLoadHardening">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option]>, NegFlag<SetFalse>, BothFlags<[CoreOption]>>,
  Group<m_Group>;
def mlvi_hardening : Flag<["-"], "mlvi-hardening">, Group<m_Group>, Flags<[CoreOption,NoXarchOption]>,
  HelpText<"Enable all mitigations for Load Value Injection (LVI)">;
def mno_lvi_hardening : Flag<["-"], "mno-lvi-hardening">, Group<m_Group>, Flags<[CoreOption,NoXarchOption]>,
  HelpText<"Disable mitigations for Load Value Injection (LVI)">;
def mlvi_cfi : Flag<["-"], "mlvi-cfi">, Group<m_Group>, Flags<[CoreOption,NoXarchOption]>,
  HelpText<"Enable only control-flow mitigations for Load Value Injection (LVI)">;
def mno_lvi_cfi : Flag<["-"], "mno-lvi-cfi">, Group<m_Group>, Flags<[CoreOption,NoXarchOption]>,
  HelpText<"Disable control-flow mitigations for Load Value Injection (LVI)">;
def m_seses : Flag<["-"], "mseses">, Group<m_Group>, Flags<[CoreOption, NoXarchOption]>,
  HelpText<"Enable speculative execution side effect suppression (SESES). "
    "Includes LVI control flow integrity mitigations">;
def mno_seses : Flag<["-"], "mno-seses">, Group<m_Group>, Flags<[CoreOption, NoXarchOption]>,
  HelpText<"Disable speculative execution side effect suppression (SESES)">;

def mrelax : Flag<["-"], "mrelax">, Group<m_Group>,
  HelpText<"Enable linker relaxation">;
def mno_relax : Flag<["-"], "mno-relax">, Group<m_Group>,
  HelpText<"Disable linker relaxation">;
def msmall_data_limit_EQ : Joined<["-"], "msmall-data-limit=">, Group<m_Group>,
  Alias<G>,
  HelpText<"Put global and static data smaller than the limit into a special section">;
def msave_restore : Flag<["-"], "msave-restore">, Group<m_riscv_Features_Group>,
  HelpText<"Enable using library calls for save and restore">;
def mno_save_restore : Flag<["-"], "mno-save-restore">, Group<m_riscv_Features_Group>,
  HelpText<"Disable using library calls for save and restore">;
def mcmodel_EQ_medlow : Flag<["-"], "mcmodel=medlow">, Group<m_riscv_Features_Group>,
  Flags<[CC1Option]>, Alias<mcmodel_EQ>, AliasArgs<["small"]>,
  HelpText<"Equivalent to -mcmodel=small, compatible with RISC-V gcc.">;
def mcmodel_EQ_medany : Flag<["-"], "mcmodel=medany">, Group<m_riscv_Features_Group>,
  Flags<[CC1Option]>, Alias<mcmodel_EQ>, AliasArgs<["medium"]>,
  HelpText<"Equivalent to -mcmodel=medium, compatible with RISC-V gcc.">;
def menable_experimental_extensions : Flag<["-"], "menable-experimental-extensions">, Group<m_Group>,
  HelpText<"Enable use of experimental RISC-V extensions.">;

def munaligned_access : Flag<["-"], "munaligned-access">, Group<m_arm_Features_Group>,
  HelpText<"Allow memory accesses to be unaligned (AArch32/AArch64 only)">;
def mno_unaligned_access : Flag<["-"], "mno-unaligned-access">, Group<m_arm_Features_Group>,
  HelpText<"Force all memory accesses to be aligned (AArch32/AArch64 only)">;
def mstrict_align : Flag<["-"], "mstrict-align">, Alias<mno_unaligned_access>, Flags<[CC1Option,HelpHidden]>,
  HelpText<"Force all memory accesses to be aligned (same as mno-unaligned-access)">;
def mno_thumb : Flag<["-"], "mno-thumb">, Group<m_arm_Features_Group>;
def mrestrict_it: Flag<["-"], "mrestrict-it">, Group<m_arm_Features_Group>,
  HelpText<"Disallow generation of complex IT blocks.">;
def mno_restrict_it: Flag<["-"], "mno-restrict-it">, Group<m_arm_Features_Group>,
  HelpText<"Allow generation of complex IT blocks.">;
def marm : Flag<["-"], "marm">, Alias<mno_thumb>;
def ffixed_r9 : Flag<["-"], "ffixed-r9">, Group<m_arm_Features_Group>,
  HelpText<"Reserve the r9 register (ARM only)">;
def mno_movt : Flag<["-"], "mno-movt">, Group<m_arm_Features_Group>,
  HelpText<"Disallow use of movt/movw pairs (ARM only)">;
def mcrc : Flag<["-"], "mcrc">, Group<m_Group>,
  HelpText<"Allow use of CRC instructions (ARM/Mips only)">;
def mnocrc : Flag<["-"], "mnocrc">, Group<m_arm_Features_Group>,
  HelpText<"Disallow use of CRC instructions (ARM only)">;
def mno_neg_immediates: Flag<["-"], "mno-neg-immediates">, Group<m_arm_Features_Group>,
  HelpText<"Disallow converting instructions with negative immediates to their negation or inversion.">;
def mcmse : Flag<["-"], "mcmse">, Group<m_arm_Features_Group>,
  Flags<[NoXarchOption,CC1Option]>,
  HelpText<"Allow use of CMSE (Armv8-M Security Extensions)">,
  MarshallingInfoFlag<LangOpts<"Cmse">>;
def ForceAAPCSBitfieldLoad : Flag<["-"], "faapcs-bitfield-load">, Group<m_arm_Features_Group>,
  Flags<[NoXarchOption,CC1Option]>,
  HelpText<"Follows the AAPCS standard that all volatile bit-field write generates at least one load. (ARM only).">,
  MarshallingInfoFlag<CodeGenOpts<"ForceAAPCSBitfieldLoad">>;
defm aapcs_bitfield_width : BoolOption<"f", "aapcs-bitfield-width",
  CodeGenOpts<"AAPCSBitfieldWidth">, DefaultTrue,
  NegFlag<SetFalse, [], "Do not follow">, PosFlag<SetTrue, [], "Follow">,
  BothFlags<[NoXarchOption, CC1Option], " the AAPCS standard requirement stating that"
            " volatile bit-field width is dictated by the field container type. (ARM only).">>,
  Group<m_arm_Features_Group>;
def mframe_chain : Joined<["-"], "mframe-chain=">,
  Group<m_arm_Features_Group>, Values<"none,aapcs,aapcs+leaf">,
  HelpText<"Select the frame chain model used to emit frame records (Arm only).">;
def mgeneral_regs_only : Flag<["-"], "mgeneral-regs-only">, Group<m_Group>,
  HelpText<"Generate code which only uses the general purpose registers (AArch64/x86 only)">;
def mfix_cmse_cve_2021_35465 : Flag<["-"], "mfix-cmse-cve-2021-35465">,
  Group<m_arm_Features_Group>,
  HelpText<"Work around VLLDM erratum CVE-2021-35465 (ARM only)">;
def mno_fix_cmse_cve_2021_35465 : Flag<["-"], "mno-fix-cmse-cve-2021-35465">,
  Group<m_arm_Features_Group>,
  HelpText<"Don't work around VLLDM erratum CVE-2021-35465 (ARM only)">;
def mfix_cortex_a57_aes_1742098 : Flag<["-"], "mfix-cortex-a57-aes-1742098">,
  Group<m_arm_Features_Group>,
  HelpText<"Work around Cortex-A57 Erratum 1742098 (ARM only)">;
def mno_fix_cortex_a57_aes_1742098 : Flag<["-"], "mno-fix-cortex-a57-aes-1742098">,
  Group<m_arm_Features_Group>,
  HelpText<"Don't work around Cortex-A57 Erratum 1742098 (ARM only)">;
def mfix_cortex_a72_aes_1655431 : Flag<["-"], "mfix-cortex-a72-aes-1655431">,
  Group<m_arm_Features_Group>,
  HelpText<"Work around Cortex-A72 Erratum 1655431 (ARM only)">,
  Alias<mfix_cortex_a57_aes_1742098>;
def mno_fix_cortex_a72_aes_1655431 : Flag<["-"], "mno-fix-cortex-a72-aes-1655431">,
  Group<m_arm_Features_Group>,
  HelpText<"Don't work around Cortex-A72 Erratum 1655431 (ARM only)">,
  Alias<mno_fix_cortex_a57_aes_1742098>;
def mfix_cortex_a53_835769 : Flag<["-"], "mfix-cortex-a53-835769">,
  Group<m_aarch64_Features_Group>,
  HelpText<"Workaround Cortex-A53 erratum 835769 (AArch64 only)">;
def mno_fix_cortex_a53_835769 : Flag<["-"], "mno-fix-cortex-a53-835769">,
  Group<m_aarch64_Features_Group>,
  HelpText<"Don't workaround Cortex-A53 erratum 835769 (AArch64 only)">;
def mmark_bti_property : Flag<["-"], "mmark-bti-property">,
  Group<m_aarch64_Features_Group>,
  HelpText<"Add .note.gnu.property with BTI to assembly files (AArch64 only)">;
def mno_bti_at_return_twice : Flag<["-"], "mno-bti-at-return-twice">,
  Group<m_arm_Features_Group>,
  HelpText<"Do not add a BTI instruction after a setjmp or other"
           " return-twice construct (Arm/AArch64 only)">;

foreach i = {1-31} in
  def ffixed_x#i : Flag<["-"], "ffixed-x"#i>, Group<m_Group>,
    HelpText<"Reserve the x"#i#" register (AArch64/RISC-V only)">;

foreach i = {8-15,18} in
  def fcall_saved_x#i : Flag<["-"], "fcall-saved-x"#i>, Group<m_aarch64_Features_Group>,
    HelpText<"Make the x"#i#" register call-saved (AArch64 only)">;

def msve_vector_bits_EQ : Joined<["-"], "msve-vector-bits=">, Group<m_aarch64_Features_Group>,
  HelpText<"Specify the size in bits of an SVE vector register. Defaults to the"
           " vector length agnostic value of \"scalable\". (AArch64 only)">;

def mvscale_min_EQ : Joined<["-"], "mvscale-min=">,
  Group<m_aarch64_Features_Group>, Flags<[NoXarchOption,CC1Option]>,
  HelpText<"Specify the vscale minimum. Defaults to \"1\". (AArch64 only)">,
  MarshallingInfoInt<LangOpts<"VScaleMin">>;
def mvscale_max_EQ : Joined<["-"], "mvscale-max=">,
  Group<m_aarch64_Features_Group>, Flags<[NoXarchOption,CC1Option]>,
  HelpText<"Specify the vscale maximum. Defaults to the"
           " vector length agnostic value of \"0\". (AArch64 only)">,
  MarshallingInfoInt<LangOpts<"VScaleMax">>;

def msign_return_address_EQ : Joined<["-"], "msign-return-address=">,
  Flags<[CC1Option]>, Group<m_Group>, Values<"none,all,non-leaf">,
  HelpText<"Select return address signing scope">;
def mbranch_protection_EQ : Joined<["-"], "mbranch-protection=">,
  Group<m_Group>,
  HelpText<"Enforce targets of indirect branches and function returns">;

def mharden_sls_EQ : Joined<["-"], "mharden-sls=">,
  HelpText<"Select straight-line speculation hardening scope (ARM/AArch64/X86"
           " only). <arg> must be: all, none, retbr(ARM/AArch64),"
           " blr(ARM/AArch64), comdat(ARM/AArch64), nocomdat(ARM/AArch64),"
           " return(X86), indirect-jmp(X86)">;

def msimd128 : Flag<["-"], "msimd128">, Group<m_wasm_Features_Group>;
def mno_simd128 : Flag<["-"], "mno-simd128">, Group<m_wasm_Features_Group>;
def mrelaxed_simd : Flag<["-"], "mrelaxed-simd">, Group<m_wasm_Features_Group>;
def mno_relaxed_simd : Flag<["-"], "mno-relaxed-simd">, Group<m_wasm_Features_Group>;
def mnontrapping_fptoint : Flag<["-"], "mnontrapping-fptoint">, Group<m_wasm_Features_Group>;
def mno_nontrapping_fptoint : Flag<["-"], "mno-nontrapping-fptoint">, Group<m_wasm_Features_Group>;
def msign_ext : Flag<["-"], "msign-ext">, Group<m_wasm_Features_Group>;
def mno_sign_ext : Flag<["-"], "mno-sign-ext">, Group<m_wasm_Features_Group>;
def mexception_handing : Flag<["-"], "mexception-handling">, Group<m_wasm_Features_Group>;
def mno_exception_handing : Flag<["-"], "mno-exception-handling">, Group<m_wasm_Features_Group>;
def matomics : Flag<["-"], "matomics">, Group<m_wasm_Features_Group>;
def mno_atomics : Flag<["-"], "mno-atomics">, Group<m_wasm_Features_Group>;
def mbulk_memory : Flag<["-"], "mbulk-memory">, Group<m_wasm_Features_Group>;
def mno_bulk_memory : Flag<["-"], "mno-bulk-memory">, Group<m_wasm_Features_Group>;
def mmutable_globals : Flag<["-"], "mmutable-globals">, Group<m_wasm_Features_Group>;
def mno_mutable_globals : Flag<["-"], "mno-mutable-globals">, Group<m_wasm_Features_Group>;
def mmultivalue : Flag<["-"], "mmultivalue">, Group<m_wasm_Features_Group>;
def mno_multivalue : Flag<["-"], "mno-multivalue">, Group<m_wasm_Features_Group>;
def mtail_call : Flag<["-"], "mtail-call">, Group<m_wasm_Features_Group>;
def mno_tail_call : Flag<["-"], "mno-tail-call">, Group<m_wasm_Features_Group>;
def mreference_types : Flag<["-"], "mreference-types">, Group<m_wasm_Features_Group>;
def mno_reference_types : Flag<["-"], "mno-reference-types">, Group<m_wasm_Features_Group>;
def mextended_const : Flag<["-"], "mextended-const">, Group<m_wasm_Features_Group>;
def mno_extended_const : Flag<["-"], "mno-extended-const">, Group<m_wasm_Features_Group>;
def mexec_model_EQ : Joined<["-"], "mexec-model=">, Group<m_wasm_Features_Driver_Group>,
                     Values<"command,reactor">,
                     HelpText<"Execution model (WebAssembly only)">;

defm amdgpu_ieee : BoolOption<"m", "amdgpu-ieee",
  CodeGenOpts<"EmitIEEENaNCompliantInsts">, DefaultTrue,
  PosFlag<SetTrue, [], "Sets the IEEE bit in the expected default floating point "
  " mode register. Floating point opcodes that support exception flag "
  "gathering quiet and propagate signaling NaN inputs per IEEE 754-2008. "
  "This option changes the ABI. (AMDGPU only)">,
  NegFlag<SetFalse, [CC1Option]>>, Group<m_Group>;

def mcode_object_version_EQ : Joined<["-"], "mcode-object-version=">, Group<m_Group>,
  HelpText<"Specify code object ABI version. Defaults to 4. (AMDGPU only)">,
  Flags<[CC1Option]>,
  Values<"none,2,3,4,5">,
  NormalizedValuesScope<"TargetOptions">,
  NormalizedValues<["COV_None", "COV_2", "COV_3", "COV_4", "COV_5"]>,
  MarshallingInfoEnum<TargetOpts<"CodeObjectVersion">, "COV_4">;

defm code_object_v3_legacy : SimpleMFlag<"code-object-v3",
  "Legacy option to specify code object ABI V3",
  "Legacy option to specify code object ABI V2",
  " (AMDGPU only)">;
defm cumode : SimpleMFlag<"cumode",
  "Specify CU wavefront", "Specify WGP wavefront",
  " execution mode (AMDGPU only)", m_amdgpu_Features_Group>;
defm tgsplit : SimpleMFlag<"tgsplit", "Enable", "Disable",
  " threadgroup split execution mode (AMDGPU only)", m_amdgpu_Features_Group>;
defm wavefrontsize64 : SimpleMFlag<"wavefrontsize64",
  "Specify wavefront size 64", "Specify wavefront size 32",
  " mode (AMDGPU only)">;

defm unsafe_fp_atomics : BoolOption<"m", "unsafe-fp-atomics",
  TargetOpts<"AllowAMDGPUUnsafeFPAtomics">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable unsafe floating point atomic instructions (AMDGPU only)">,
  NegFlag<SetFalse>>, Group<m_Group>;

def faltivec : Flag<["-"], "faltivec">, Group<f_Group>, Flags<[NoXarchOption]>;
def fno_altivec : Flag<["-"], "fno-altivec">, Group<f_Group>, Flags<[NoXarchOption]>;
def maltivec : Flag<["-"], "maltivec">, Group<m_ppc_Features_Group>;
def mno_altivec : Flag<["-"], "mno-altivec">, Group<m_ppc_Features_Group>;
def mpcrel: Flag<["-"], "mpcrel">, Group<m_ppc_Features_Group>;
def mno_pcrel: Flag<["-"], "mno-pcrel">, Group<m_ppc_Features_Group>;
def mprefixed: Flag<["-"], "mprefixed">, Group<m_ppc_Features_Group>;
def mno_prefixed: Flag<["-"], "mno-prefixed">, Group<m_ppc_Features_Group>;
def mspe : Flag<["-"], "mspe">, Group<m_ppc_Features_Group>;
def mno_spe : Flag<["-"], "mno-spe">, Group<m_ppc_Features_Group>;
def mefpu2 : Flag<["-"], "mefpu2">, Group<m_ppc_Features_Group>;
def mabi_EQ_vec_extabi : Flag<["-"], "mabi=vec-extabi">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Enable the extended Altivec ABI on AIX (AIX only). Uses volatile and nonvolatile vector registers">,
  MarshallingInfoFlag<LangOpts<"EnableAIXExtendedAltivecABI">>;
def mabi_EQ_vec_default : Flag<["-"], "mabi=vec-default">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Enable the default Altivec ABI on AIX (AIX only). Uses only volatile vector registers.">;
def mvsx : Flag<["-"], "mvsx">, Group<m_ppc_Features_Group>;
def mno_vsx : Flag<["-"], "mno-vsx">, Group<m_ppc_Features_Group>;
def msecure_plt : Flag<["-"], "msecure-plt">, Group<m_ppc_Features_Group>;
def mpower8_vector : Flag<["-"], "mpower8-vector">,
    Group<m_ppc_Features_Group>;
def mno_power8_vector : Flag<["-"], "mno-power8-vector">,
    Group<m_ppc_Features_Group>;
def mpower9_vector : Flag<["-"], "mpower9-vector">,
    Group<m_ppc_Features_Group>;
def mno_power9_vector : Flag<["-"], "mno-power9-vector">,
    Group<m_ppc_Features_Group>;
def mpower10_vector : Flag<["-"], "mpower10-vector">,
    Group<m_ppc_Features_Group>;
def mno_power10_vector : Flag<["-"], "mno-power10-vector">,
    Group<m_ppc_Features_Group>;
def mpower8_crypto : Flag<["-"], "mcrypto">,
    Group<m_ppc_Features_Group>;
def mnopower8_crypto : Flag<["-"], "mno-crypto">,
    Group<m_ppc_Features_Group>;
def mdirect_move : Flag<["-"], "mdirect-move">,
    Group<m_ppc_Features_Group>;
def mnodirect_move : Flag<["-"], "mno-direct-move">,
    Group<m_ppc_Features_Group>;
def mpaired_vector_memops: Flag<["-"], "mpaired-vector-memops">,
    Group<m_ppc_Features_Group>;
def mnopaired_vector_memops: Flag<["-"], "mno-paired-vector-memops">,
    Group<m_ppc_Features_Group>;
def mhtm : Flag<["-"], "mhtm">, Group<m_ppc_Features_Group>;
def mno_htm : Flag<["-"], "mno-htm">, Group<m_ppc_Features_Group>;
def mfprnd : Flag<["-"], "mfprnd">, Group<m_ppc_Features_Group>;
def mno_fprnd : Flag<["-"], "mno-fprnd">, Group<m_ppc_Features_Group>;
def mcmpb : Flag<["-"], "mcmpb">, Group<m_ppc_Features_Group>;
def mno_cmpb : Flag<["-"], "mno-cmpb">, Group<m_ppc_Features_Group>;
def misel : Flag<["-"], "misel">, Group<m_ppc_Features_Group>;
def mno_isel : Flag<["-"], "mno-isel">, Group<m_ppc_Features_Group>;
def mmfocrf : Flag<["-"], "mmfocrf">, Group<m_ppc_Features_Group>;
def mmfcrf : Flag<["-"], "mmfcrf">, Alias<mmfocrf>;
def mno_mfocrf : Flag<["-"], "mno-mfocrf">, Group<m_ppc_Features_Group>;
def mno_mfcrf : Flag<["-"], "mno-mfcrf">, Alias<mno_mfocrf>;
def mpopcntd : Flag<["-"], "mpopcntd">, Group<m_ppc_Features_Group>;
def mno_popcntd : Flag<["-"], "mno-popcntd">, Group<m_ppc_Features_Group>;
def mcrbits : Flag<["-"], "mcrbits">, Group<m_ppc_Features_Group>;
def mno_crbits : Flag<["-"], "mno-crbits">, Group<m_ppc_Features_Group>;
def minvariant_function_descriptors :
  Flag<["-"], "minvariant-function-descriptors">, Group<m_ppc_Features_Group>;
def mno_invariant_function_descriptors :
  Flag<["-"], "mno-invariant-function-descriptors">,
  Group<m_ppc_Features_Group>;
def mfloat128: Flag<["-"], "mfloat128">,
    Group<m_ppc_Features_Group>;
def mno_float128 : Flag<["-"], "mno-float128">,
    Group<m_ppc_Features_Group>;
def mlongcall: Flag<["-"], "mlongcall">,
    Group<m_ppc_Features_Group>;
def mno_longcall : Flag<["-"], "mno-longcall">,
    Group<m_ppc_Features_Group>;
def mmma: Flag<["-"], "mmma">, Group<m_ppc_Features_Group>;
def mno_mma: Flag<["-"], "mno-mma">, Group<m_ppc_Features_Group>;
def mrop_protect : Flag<["-"], "mrop-protect">,
    Group<m_ppc_Features_Group>;
def mprivileged : Flag<["-"], "mprivileged">,
    Group<m_ppc_Features_Group>;
def maix_struct_return : Flag<["-"], "maix-struct-return">,
  Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Return all structs in memory (PPC32 only)">;
def msvr4_struct_return : Flag<["-"], "msvr4-struct-return">,
  Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Return small structs in registers (PPC32 only)">;

def mvx : Flag<["-"], "mvx">, Group<m_Group>;
def mno_vx : Flag<["-"], "mno-vx">, Group<m_Group>;

defm zvector : BoolFOption<"zvector",
  LangOpts<"ZVector">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Enable System z vector language extension">,
  NegFlag<SetFalse>>;
def mzvector : Flag<["-"], "mzvector">, Alias<fzvector>;
def mno_zvector : Flag<["-"], "mno-zvector">, Alias<fno_zvector>;

def mignore_xcoff_visibility : Flag<["-"], "mignore-xcoff-visibility">, Group<m_Group>,
HelpText<"Not emit the visibility attribute for asm in AIX OS or give all symbols 'unspecified' visibility in XCOFF object file">,
  Flags<[CC1Option]>;
defm backchain : BoolOption<"m", "backchain",
  CodeGenOpts<"Backchain">, DefaultFalse,
  PosFlag<SetTrue, [], "Link stack frames through backchain on System Z">,
  NegFlag<SetFalse>, BothFlags<[NoXarchOption,CC1Option]>>, Group<m_Group>;

def mno_warn_nonportable_cfstrings : Flag<["-"], "mno-warn-nonportable-cfstrings">, Group<m_Group>;
def mno_omit_leaf_frame_pointer : Flag<["-"], "mno-omit-leaf-frame-pointer">, Group<m_Group>;
def momit_leaf_frame_pointer : Flag<["-"], "momit-leaf-frame-pointer">, Group<m_Group>,
  HelpText<"Omit frame pointer setup for leaf functions">;
def moslib_EQ : Joined<["-"], "moslib=">, Group<m_Group>;
def mpascal_strings : Flag<["-"], "mpascal-strings">, Alias<fpascal_strings>;
def mred_zone : Flag<["-"], "mred-zone">, Group<m_Group>;
def mtls_direct_seg_refs : Flag<["-"], "mtls-direct-seg-refs">, Group<m_Group>,
  HelpText<"Enable direct TLS access through segment registers (default)">;
def mregparm_EQ : Joined<["-"], "mregparm=">, Group<m_Group>;
def mrelax_all : Flag<["-"], "mrelax-all">, Group<m_Group>, Flags<[CC1Option,CC1AsOption]>,
  HelpText<"(integrated-as) Relax all machine instructions">,
  MarshallingInfoFlag<CodeGenOpts<"RelaxAll">>;
def mincremental_linker_compatible : Flag<["-"], "mincremental-linker-compatible">, Group<m_Group>,
  Flags<[CC1Option,CC1AsOption]>,
  HelpText<"(integrated-as) Emit an object file which can be used with an incremental linker">,
  MarshallingInfoFlag<CodeGenOpts<"IncrementalLinkerCompatible">>;
def mno_incremental_linker_compatible : Flag<["-"], "mno-incremental-linker-compatible">, Group<m_Group>,
  HelpText<"(integrated-as) Emit an object file which cannot be used with an incremental linker">;
def mrtd : Flag<["-"], "mrtd">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Make StdCall calling convention the default">;
def msmall_data_threshold_EQ : Joined <["-"], "msmall-data-threshold=">,
  Group<m_Group>, Alias<G>;
def msoft_float : Flag<["-"], "msoft-float">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Use software floating point">,
  MarshallingInfoFlag<CodeGenOpts<"SoftFloat">>;
def moutline_atomics : Flag<["-"], "moutline-atomics">, Group<f_clang_Group>, Flags<[CC1Option]>,
  HelpText<"Generate local calls to out-of-line atomic operations">;
def mno_outline_atomics : Flag<["-"], "mno-outline-atomics">, Group<f_clang_Group>, Flags<[CC1Option]>,
  HelpText<"Don't generate local calls to out-of-line atomic operations">;
def mno_implicit_float : Flag<["-"], "mno-implicit-float">, Group<m_Group>,
  HelpText<"Don't generate implicit floating point instructions">;
def mimplicit_float : Flag<["-"], "mimplicit-float">, Group<m_Group>;
def mrecip : Flag<["-"], "mrecip">, Group<m_Group>;
def mrecip_EQ : CommaJoined<["-"], "mrecip=">, Group<m_Group>, Flags<[CC1Option]>,
  MarshallingInfoStringVector<CodeGenOpts<"Reciprocals">>;
def mprefer_vector_width_EQ : Joined<["-"], "mprefer-vector-width=">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Specifies preferred vector width for auto-vectorization. Defaults to 'none' which allows target specific decisions.">,
  MarshallingInfoString<CodeGenOpts<"PreferVectorWidth">>;
def mstack_protector_guard_EQ : Joined<["-"], "mstack-protector-guard=">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Use the given guard (global, tls) for addressing the stack-protector guard">,
  MarshallingInfoString<CodeGenOpts<"StackProtectorGuard">>;
def mstack_protector_guard_offset_EQ : Joined<["-"], "mstack-protector-guard-offset=">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Use the given offset for addressing the stack-protector guard">,
  MarshallingInfoInt<CodeGenOpts<"StackProtectorGuardOffset">, "INT_MAX", "int">;
def mstack_protector_guard_symbol_EQ : Joined<["-"], "mstack-protector-guard-symbol=">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Use the given symbol for addressing the stack-protector guard">,
  MarshallingInfoString<CodeGenOpts<"StackProtectorGuardSymbol">>;
def mstack_protector_guard_reg_EQ : Joined<["-"], "mstack-protector-guard-reg=">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Use the given reg for addressing the stack-protector guard">,
  MarshallingInfoString<CodeGenOpts<"StackProtectorGuardReg">>;
def mfentry : Flag<["-"], "mfentry">, HelpText<"Insert calls to fentry at function entry (x86/SystemZ only)">,
  Flags<[CC1Option]>, Group<m_Group>,
  MarshallingInfoFlag<CodeGenOpts<"CallFEntry">>;
def mnop_mcount : Flag<["-"], "mnop-mcount">, HelpText<"Generate mcount/__fentry__ calls as nops. To activate they need to be patched in.">,
  Flags<[CC1Option]>, Group<m_Group>,
  MarshallingInfoFlag<CodeGenOpts<"MNopMCount">>;
def mrecord_mcount : Flag<["-"], "mrecord-mcount">, HelpText<"Generate a __mcount_loc section entry for each __fentry__ call.">,
  Flags<[CC1Option]>, Group<m_Group>,
  MarshallingInfoFlag<CodeGenOpts<"RecordMCount">>;
def mpacked_stack : Flag<["-"], "mpacked-stack">, HelpText<"Use packed stack layout (SystemZ only).">,
  Flags<[CC1Option]>, Group<m_Group>,
  MarshallingInfoFlag<CodeGenOpts<"PackedStack">>;
def mno_packed_stack : Flag<["-"], "mno-packed-stack">, Flags<[CC1Option]>, Group<m_Group>;
def mips16 : Flag<["-"], "mips16">, Group<m_mips_Features_Group>;
def mno_mips16 : Flag<["-"], "mno-mips16">, Group<m_mips_Features_Group>;
def mmicromips : Flag<["-"], "mmicromips">, Group<m_mips_Features_Group>;
def mno_micromips : Flag<["-"], "mno-micromips">, Group<m_mips_Features_Group>;
def mxgot : Flag<["-"], "mxgot">, Group<m_mips_Features_Group>;
def mno_xgot : Flag<["-"], "mno-xgot">, Group<m_mips_Features_Group>;
def mldc1_sdc1 : Flag<["-"], "mldc1-sdc1">, Group<m_mips_Features_Group>;
def mno_ldc1_sdc1 : Flag<["-"], "mno-ldc1-sdc1">, Group<m_mips_Features_Group>;
def mcheck_zero_division : Flag<["-"], "mcheck-zero-division">,
                           Group<m_mips_Features_Group>;
def mno_check_zero_division : Flag<["-"], "mno-check-zero-division">,
                              Group<m_mips_Features_Group>;
def mfix4300 : Flag<["-"], "mfix4300">, Group<m_mips_Features_Group>;
def mcompact_branches_EQ : Joined<["-"], "mcompact-branches=">,
                           Group<m_mips_Features_Group>;
def mbranch_likely : Flag<["-"], "mbranch-likely">, Group<m_Group>,
  IgnoredGCCCompat;
def mno_branch_likely : Flag<["-"], "mno-branch-likely">, Group<m_Group>,
  IgnoredGCCCompat;
def mindirect_jump_EQ : Joined<["-"], "mindirect-jump=">,
  Group<m_mips_Features_Group>,
  HelpText<"Change indirect jump instructions to inhibit speculation">;
def mdsp : Flag<["-"], "mdsp">, Group<m_mips_Features_Group>;
def mno_dsp : Flag<["-"], "mno-dsp">, Group<m_mips_Features_Group>;
def mdspr2 : Flag<["-"], "mdspr2">, Group<m_mips_Features_Group>;
def mno_dspr2 : Flag<["-"], "mno-dspr2">, Group<m_mips_Features_Group>;
def msingle_float : Flag<["-"], "msingle-float">, Group<m_mips_Features_Group>;
def mdouble_float : Flag<["-"], "mdouble-float">, Group<m_mips_Features_Group>;
def mmadd4 : Flag<["-"], "mmadd4">, Group<m_mips_Features_Group>,
  HelpText<"Enable the generation of 4-operand madd.s, madd.d and related instructions.">;
def mno_madd4 : Flag<["-"], "mno-madd4">, Group<m_mips_Features_Group>,
  HelpText<"Disable the generation of 4-operand madd.s, madd.d and related instructions.">;
def mmsa : Flag<["-"], "mmsa">, Group<m_mips_Features_Group>,
  HelpText<"Enable MSA ASE (MIPS only)">;
def mno_msa : Flag<["-"], "mno-msa">, Group<m_mips_Features_Group>,
  HelpText<"Disable MSA ASE (MIPS only)">;
def mmt : Flag<["-"], "mmt">, Group<m_mips_Features_Group>,
  HelpText<"Enable MT ASE (MIPS only)">;
def mno_mt : Flag<["-"], "mno-mt">, Group<m_mips_Features_Group>,
  HelpText<"Disable MT ASE (MIPS only)">;
def mfp64 : Flag<["-"], "mfp64">, Group<m_mips_Features_Group>,
  HelpText<"Use 64-bit floating point registers (MIPS only)">;
def mfp32 : Flag<["-"], "mfp32">, Group<m_mips_Features_Group>,
  HelpText<"Use 32-bit floating point registers (MIPS only)">;
def mgpopt : Flag<["-"], "mgpopt">, Group<m_mips_Features_Group>,
  HelpText<"Use GP relative accesses for symbols known to be in a small"
           " data section (MIPS)">;
def mno_gpopt : Flag<["-"], "mno-gpopt">, Group<m_mips_Features_Group>,
  HelpText<"Do not use GP relative accesses for symbols known to be in a small"
           " data section (MIPS)">;
def mlocal_sdata : Flag<["-"], "mlocal-sdata">,
  Group<m_mips_Features_Group>,
  HelpText<"Extend the -G behaviour to object local data (MIPS)">;
def mno_local_sdata : Flag<["-"], "mno-local-sdata">,
  Group<m_mips_Features_Group>,
  HelpText<"Do not extend the -G behaviour to object local data (MIPS)">;
def mextern_sdata : Flag<["-"], "mextern-sdata">,
  Group<m_mips_Features_Group>,
  HelpText<"Assume that externally defined data is in the small data if it"
           " meets the -G <size> threshold (MIPS)">;
def mno_extern_sdata : Flag<["-"], "mno-extern-sdata">,
  Group<m_mips_Features_Group>,
  HelpText<"Do not assume that externally defined data is in the small data if"
           " it meets the -G <size> threshold (MIPS)">;
def membedded_data : Flag<["-"], "membedded-data">,
  Group<m_mips_Features_Group>,
  HelpText<"Place constants in the .rodata section instead of the .sdata "
           "section even if they meet the -G <size> threshold (MIPS)">;
def mno_embedded_data : Flag<["-"], "mno-embedded-data">,
  Group<m_mips_Features_Group>,
  HelpText<"Do not place constants in the .rodata section instead of the "
           ".sdata if they meet the -G <size> threshold (MIPS)">;
def mnan_EQ : Joined<["-"], "mnan=">, Group<m_mips_Features_Group>;
def mabs_EQ : Joined<["-"], "mabs=">, Group<m_mips_Features_Group>;
def mabicalls : Flag<["-"], "mabicalls">, Group<m_mips_Features_Group>,
  HelpText<"Enable SVR4-style position-independent code (Mips only)">;
def mno_abicalls : Flag<["-"], "mno-abicalls">, Group<m_mips_Features_Group>,
  HelpText<"Disable SVR4-style position-independent code (Mips only)">;
def mno_crc : Flag<["-"], "mno-crc">, Group<m_mips_Features_Group>,
  HelpText<"Disallow use of CRC instructions (Mips only)">;
def mvirt : Flag<["-"], "mvirt">, Group<m_mips_Features_Group>;
def mno_virt : Flag<["-"], "mno-virt">, Group<m_mips_Features_Group>;
def mginv : Flag<["-"], "mginv">, Group<m_mips_Features_Group>;
def mno_ginv : Flag<["-"], "mno-ginv">, Group<m_mips_Features_Group>;
def mips1 : Flag<["-"], "mips1">,
  Alias<march_EQ>, AliasArgs<["mips1"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips1">, Flags<[HelpHidden]>;
def mips2 : Flag<["-"], "mips2">,
  Alias<march_EQ>, AliasArgs<["mips2"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips2">, Flags<[HelpHidden]>;
def mips3 : Flag<["-"], "mips3">,
  Alias<march_EQ>, AliasArgs<["mips3"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips3">, Flags<[HelpHidden]>;
def mips4 : Flag<["-"], "mips4">,
  Alias<march_EQ>, AliasArgs<["mips4"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips4">, Flags<[HelpHidden]>;
def mips5 : Flag<["-"], "mips5">,
  Alias<march_EQ>, AliasArgs<["mips5"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips5">, Flags<[HelpHidden]>;
def mips32 : Flag<["-"], "mips32">,
  Alias<march_EQ>, AliasArgs<["mips32"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips32">, Flags<[HelpHidden]>;
def mips32r2 : Flag<["-"], "mips32r2">,
  Alias<march_EQ>, AliasArgs<["mips32r2"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips32r2">, Flags<[HelpHidden]>;
def mips32r3 : Flag<["-"], "mips32r3">,
  Alias<march_EQ>, AliasArgs<["mips32r3"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips32r3">, Flags<[HelpHidden]>;
def mips32r5 : Flag<["-"], "mips32r5">,
  Alias<march_EQ>, AliasArgs<["mips32r5"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips32r5">, Flags<[HelpHidden]>;
def mips32r6 : Flag<["-"], "mips32r6">,
  Alias<march_EQ>, AliasArgs<["mips32r6"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips32r6">, Flags<[HelpHidden]>;
def mips64 : Flag<["-"], "mips64">,
  Alias<march_EQ>, AliasArgs<["mips64"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips64">, Flags<[HelpHidden]>;
def mips64r2 : Flag<["-"], "mips64r2">,
  Alias<march_EQ>, AliasArgs<["mips64r2"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips64r2">, Flags<[HelpHidden]>;
def mips64r3 : Flag<["-"], "mips64r3">,
  Alias<march_EQ>, AliasArgs<["mips64r3"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips64r3">, Flags<[HelpHidden]>;
def mips64r5 : Flag<["-"], "mips64r5">,
  Alias<march_EQ>, AliasArgs<["mips64r5"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips64r5">, Flags<[HelpHidden]>;
def mips64r6 : Flag<["-"], "mips64r6">,
  Alias<march_EQ>, AliasArgs<["mips64r6"]>, Group<m_mips_Features_Group>,
  HelpText<"Equivalent to -march=mips64r6">, Flags<[HelpHidden]>;
def mfpxx : Flag<["-"], "mfpxx">, Group<m_mips_Features_Group>,
  HelpText<"Avoid FPU mode dependent operations when used with the O32 ABI">,
  Flags<[HelpHidden]>;
def modd_spreg : Flag<["-"], "modd-spreg">, Group<m_mips_Features_Group>,
  HelpText<"Enable odd single-precision floating point registers">,
  Flags<[HelpHidden]>;
def mno_odd_spreg : Flag<["-"], "mno-odd-spreg">, Group<m_mips_Features_Group>,
  HelpText<"Disable odd single-precision floating point registers">,
  Flags<[HelpHidden]>;
def mrelax_pic_calls : Flag<["-"], "mrelax-pic-calls">,
  Group<m_mips_Features_Group>,
  HelpText<"Produce relaxation hints for linkers to try optimizing PIC "
           "call sequences into direct calls (MIPS only)">, Flags<[HelpHidden]>;
def mno_relax_pic_calls : Flag<["-"], "mno-relax-pic-calls">,
  Group<m_mips_Features_Group>,
  HelpText<"Do not produce relaxation hints for linkers to try optimizing PIC "
           "call sequences into direct calls (MIPS only)">, Flags<[HelpHidden]>;
def mglibc : Flag<["-"], "mglibc">, Group<m_libc_Group>, Flags<[HelpHidden]>;
def muclibc : Flag<["-"], "muclibc">, Group<m_libc_Group>, Flags<[HelpHidden]>;
def module_file_info : Flag<["-"], "module-file-info">, Flags<[NoXarchOption,CC1Option]>, Group<Action_Group>,
  HelpText<"Provide information about a particular module file">;
def mthumb : Flag<["-"], "mthumb">, Group<m_Group>;
def mtune_EQ : Joined<["-"], "mtune=">, Group<m_Group>,
  HelpText<"Only supported on X86, RISC-V and SystemZ. Otherwise accepted for compatibility with GCC.">;
def multi__module : Flag<["-"], "multi_module">;
def multiply__defined__unused : Separate<["-"], "multiply_defined_unused">;
def multiply__defined : Separate<["-"], "multiply_defined">;
def mwarn_nonportable_cfstrings : Flag<["-"], "mwarn-nonportable-cfstrings">, Group<m_Group>;
def canonical_prefixes : Flag<["-"], "canonical-prefixes">, Flags<[HelpHidden, CoreOption]>,
  HelpText<"Use absolute paths for invoking subcommands (default)">;
def no_canonical_prefixes : Flag<["-"], "no-canonical-prefixes">, Flags<[HelpHidden, CoreOption]>,
  HelpText<"Use relative paths for invoking subcommands">;
def no_cpp_precomp : Flag<["-"], "no-cpp-precomp">, Group<clang_ignored_f_Group>;
def no_integrated_cpp : Flag<["-", "--"], "no-integrated-cpp">, Flags<[NoXarchOption]>;
def no_pedantic : Flag<["-", "--"], "no-pedantic">, Group<pedantic_Group>;
def no__dead__strip__inits__and__terms : Flag<["-"], "no_dead_strip_inits_and_terms">;
def nobuiltininc : Flag<["-"], "nobuiltininc">, Flags<[CC1Option, CoreOption]>,
  HelpText<"Disable builtin #include directories">,
  MarshallingInfoNegativeFlag<HeaderSearchOpts<"UseBuiltinIncludes">>;
def nogpuinc : Flag<["-"], "nogpuinc">, HelpText<"Do not add include paths for CUDA/HIP and"
  " do not include the default CUDA/HIP wrapper headers">;
def nohipwrapperinc : Flag<["-"], "nohipwrapperinc">,
  HelpText<"Do not include the default HIP wrapper headers and include paths">;
def : Flag<["-"], "nocudainc">, Alias<nogpuinc>;
def nogpulib : Flag<["-"], "nogpulib">, MarshallingInfoFlag<LangOpts<"NoGPULib">>,
  Flags<[CC1Option]>, HelpText<"Do not link device library for CUDA/HIP device compilation">;
def : Flag<["-"], "nocudalib">, Alias<nogpulib>;
def nodefaultlibs : Flag<["-"], "nodefaultlibs">;
def nodriverkitlib : Flag<["-"], "nodriverkitlib">;
def nofixprebinding : Flag<["-"], "nofixprebinding">;
def nolibc : Flag<["-"], "nolibc">;
def nomultidefs : Flag<["-"], "nomultidefs">;
def nopie : Flag<["-"], "nopie">;
def no_pie : Flag<["-"], "no-pie">, Alias<nopie>;
def noprebind : Flag<["-"], "noprebind">;
def noprofilelib : Flag<["-"], "noprofilelib">;
def noseglinkedit : Flag<["-"], "noseglinkedit">;
def nostartfiles : Flag<["-"], "nostartfiles">, Group<Link_Group>;
def nostdinc : Flag<["-"], "nostdinc">, Flags<[CoreOption]>;
def nostdlibinc : Flag<["-"], "nostdlibinc">;
def nostdincxx : Flag<["-"], "nostdinc++">, Flags<[CC1Option]>,
  HelpText<"Disable standard #include directories for the C++ standard library">,
  MarshallingInfoNegativeFlag<HeaderSearchOpts<"UseStandardCXXIncludes">>;
def nostdlib : Flag<["-"], "nostdlib">, Group<Link_Group>;
def nostdlibxx : Flag<["-"], "nostdlib++">;
def object : Flag<["-"], "object">;
def o : JoinedOrSeparate<["-"], "o">, Flags<[NoXarchOption, RenderAsInput,
  CC1Option, CC1AsOption, FC1Option, FlangOption]>,
  HelpText<"Write output to <file>">, MetaVarName<"<file>">,
  MarshallingInfoString<FrontendOpts<"OutputFile">>;
def object_file_name_EQ : Joined<["-"], "object-file-name=">, Flags<[CC1Option, CC1AsOption, CoreOption]>,
  HelpText<"Set the output <file> for debug infos">, MetaVarName<"<file>">,
  MarshallingInfoString<CodeGenOpts<"ObjectFilenameForDebug">>;
def object_file_name : Separate<["-"], "object-file-name">, Flags<[CC1Option, CC1AsOption, CoreOption]>,
    Alias<object_file_name_EQ>;
def pagezero__size : JoinedOrSeparate<["-"], "pagezero_size">;
def pass_exit_codes : Flag<["-", "--"], "pass-exit-codes">, Flags<[Unsupported]>;
def pedantic_errors : Flag<["-", "--"], "pedantic-errors">, Group<pedantic_Group>, Flags<[CC1Option]>,
  MarshallingInfoFlag<DiagnosticOpts<"PedanticErrors">>;
def pedantic : Flag<["-", "--"], "pedantic">, Group<pedantic_Group>, Flags<[CC1Option,FlangOption,FC1Option]>,
  HelpText<"Warn on language extensions">, MarshallingInfoFlag<DiagnosticOpts<"Pedantic">>;
def pg : Flag<["-"], "pg">, HelpText<"Enable mcount instrumentation">, Flags<[CC1Option]>,
  MarshallingInfoFlag<CodeGenOpts<"InstrumentForProfiling">>;
def pipe : Flag<["-", "--"], "pipe">,
  HelpText<"Use pipes between commands, when possible">;
def prebind__all__twolevel__modules : Flag<["-"], "prebind_all_twolevel_modules">;
def prebind : Flag<["-"], "prebind">;
def preload : Flag<["-"], "preload">;
def print_file_name_EQ : Joined<["-", "--"], "print-file-name=">,
  HelpText<"Print the full library path of <file>">, MetaVarName<"<file>">;
def print_ivar_layout : Flag<["-"], "print-ivar-layout">, Flags<[CC1Option]>,
  HelpText<"Enable Objective-C Ivar layout bitmap print trace">,
  MarshallingInfoFlag<LangOpts<"ObjCGCBitmapPrint">>;
def print_libgcc_file_name : Flag<["-", "--"], "print-libgcc-file-name">,
  HelpText<"Print the library path for the currently used compiler runtime "
           "library (\"libgcc.a\" or \"libclang_rt.builtins.*.a\")">;
def print_multi_directory : Flag<["-", "--"], "print-multi-directory">;
def print_multi_lib : Flag<["-", "--"], "print-multi-lib">;
def print_multi_os_directory : Flag<["-", "--"], "print-multi-os-directory">,
  Flags<[Unsupported]>;
def print_target_triple : Flag<["-", "--"], "print-target-triple">,
  HelpText<"Print the normalized target triple">, Flags<[FlangOption]>;
def print_effective_triple : Flag<["-", "--"], "print-effective-triple">,
  HelpText<"Print the effective target triple">, Flags<[FlangOption]>;
def print_multiarch : Flag<["-", "--"], "print-multiarch">,
  HelpText<"Print the multiarch target triple">;
def print_prog_name_EQ : Joined<["-", "--"], "print-prog-name=">,
  HelpText<"Print the full program path of <name>">, MetaVarName<"<name>">;
def print_resource_dir : Flag<["-", "--"], "print-resource-dir">,
  HelpText<"Print the resource directory pathname">;
def print_search_dirs : Flag<["-", "--"], "print-search-dirs">,
  HelpText<"Print the paths used for finding libraries and programs">;
def print_targets : Flag<["-", "--"], "print-targets">,
  HelpText<"Print the registered targets">;
def print_rocm_search_dirs : Flag<["-", "--"], "print-rocm-search-dirs">,
  HelpText<"Print the paths used for finding ROCm installation">;
def print_runtime_dir : Flag<["-", "--"], "print-runtime-dir">,
  HelpText<"Print the directory pathname containing clangs runtime libraries">;
def print_diagnostic_options : Flag<["-", "--"], "print-diagnostic-options">,
  HelpText<"Print all of Clang's warning options">;
def private__bundle : Flag<["-"], "private_bundle">;
def pthreads : Flag<["-"], "pthreads">;
defm pthread : BoolOption<"", "pthread",
  LangOpts<"POSIXThreads">, DefaultFalse,
  PosFlag<SetTrue, [], "Support POSIX threads in generated code">,
  NegFlag<SetFalse>, BothFlags<[CC1Option]>>;
def p : Flag<["-"], "p">;
def pie : Flag<["-"], "pie">, Group<Link_Group>;
def static_pie : Flag<["-"], "static-pie">, Group<Link_Group>;
def read__only__relocs : Separate<["-"], "read_only_relocs">;
def remap : Flag<["-"], "remap">;
def rewrite_objc : Flag<["-"], "rewrite-objc">, Flags<[NoXarchOption,CC1Option]>,
  HelpText<"Rewrite Objective-C source to C++">, Group<Action_Group>;
def rewrite_legacy_objc : Flag<["-"], "rewrite-legacy-objc">, Flags<[NoXarchOption]>,
  HelpText<"Rewrite Legacy Objective-C source to C++">;
def rdynamic : Flag<["-"], "rdynamic">, Group<Link_Group>;
def resource_dir : Separate<["-"], "resource-dir">,
  Flags<[NoXarchOption, CC1Option, CoreOption, HelpHidden]>,
  HelpText<"The directory which holds the compiler resource files">,
  MarshallingInfoString<HeaderSearchOpts<"ResourceDir">>;
def resource_dir_EQ : Joined<["-"], "resource-dir=">, Flags<[NoXarchOption, CoreOption]>,
  Alias<resource_dir>;
def rpath : Separate<["-"], "rpath">, Flags<[LinkerInput]>, Group<Link_Group>;
def rtlib_EQ : Joined<["-", "--"], "rtlib=">,
  HelpText<"Compiler runtime library to use">;
def frtlib_add_rpath: Flag<["-"], "frtlib-add-rpath">, Flags<[NoArgumentUnused]>,
  HelpText<"Add -rpath with architecture-specific resource directory to the linker flags">;
def fno_rtlib_add_rpath: Flag<["-"], "fno-rtlib-add-rpath">, Flags<[NoArgumentUnused]>,
  HelpText<"Do not add -rpath with architecture-specific resource directory to the linker flags">;
defm openmp_implicit_rpath: BoolFOption<"openmp-implicit-rpath",
  LangOpts<"OpenMP">,
  DefaultTrue,
  PosFlag<SetTrue, [], "Set rpath on OpenMP executables">,
  NegFlag<SetFalse>>;
def r : Flag<["-"], "r">, Flags<[LinkerInput,NoArgumentUnused]>,
        Group<Link_Group>;
def save_temps_EQ : Joined<["-", "--"], "save-temps=">, Flags<[CC1Option, FlangOption, NoXarchOption]>,
  HelpText<"Save intermediate compilation results.">;
def save_temps : Flag<["-", "--"], "save-temps">, Flags<[FlangOption, NoXarchOption]>,
  Alias<save_temps_EQ>, AliasArgs<["cwd"]>,
  HelpText<"Save intermediate compilation results">;
def save_stats_EQ : Joined<["-", "--"], "save-stats=">, Flags<[NoXarchOption]>,
  HelpText<"Save llvm statistics.">;
def save_stats : Flag<["-", "--"], "save-stats">, Flags<[NoXarchOption]>,
  Alias<save_stats_EQ>, AliasArgs<["cwd"]>,
  HelpText<"Save llvm statistics.">;
def via_file_asm : Flag<["-", "--"], "via-file-asm">, InternalDebugOpt,
  HelpText<"Write assembly to file for input to assemble jobs">;
def sectalign : MultiArg<["-"], "sectalign", 3>;
def sectcreate : MultiArg<["-"], "sectcreate", 3>;
def sectobjectsymbols : MultiArg<["-"], "sectobjectsymbols", 2>;
def sectorder : MultiArg<["-"], "sectorder", 3>;
def seg1addr : JoinedOrSeparate<["-"], "seg1addr">;
def seg__addr__table__filename : Separate<["-"], "seg_addr_table_filename">;
def seg__addr__table : Separate<["-"], "seg_addr_table">;
def segaddr : MultiArg<["-"], "segaddr", 2>;
def segcreate : MultiArg<["-"], "segcreate", 3>;
def seglinkedit : Flag<["-"], "seglinkedit">;
def segprot : MultiArg<["-"], "segprot", 3>;
def segs__read__only__addr : Separate<["-"], "segs_read_only_addr">;
def segs__read__write__addr : Separate<["-"], "segs_read_write_addr">;
def segs__read__ : Joined<["-"], "segs_read_">;
def shared_libgcc : Flag<["-"], "shared-libgcc">;
def shared : Flag<["-", "--"], "shared">, Group<Link_Group>;
def single__module : Flag<["-"], "single_module">;
def specs_EQ : Joined<["-", "--"], "specs=">, Group<Link_Group>;
def specs : Separate<["-", "--"], "specs">, Flags<[Unsupported]>;
def start_no_unused_arguments : Flag<["--"], "start-no-unused-arguments">, Flags<[CoreOption]>,
  HelpText<"Don't emit warnings about unused arguments for the following arguments">;
def static_libgcc : Flag<["-"], "static-libgcc">;
def static_libstdcxx : Flag<["-"], "static-libstdc++">;
def static : Flag<["-", "--"], "static">, Group<Link_Group>, Flags<[NoArgumentUnused]>;
def std_default_EQ : Joined<["-"], "std-default=">;
def std_EQ : Joined<["-", "--"], "std=">, Flags<[CC1Option,FlangOption,FC1Option]>,
  Group<CompileOnly_Group>, HelpText<"Language standard to compile for">,
  ValuesCode<[{
    const char *Values =
    #define LANGSTANDARD(id, name, lang, desc, features) name ","
    #define LANGSTANDARD_ALIAS(id, alias) alias ","
    #include "clang/Basic/LangStandards.def"
    ;
  }]>;
def stdlib_EQ : Joined<["-", "--"], "stdlib=">, Flags<[CC1Option]>,
  HelpText<"C++ standard library to use">, Values<"libc++,libstdc++,platform">;
def stdlibxx_isystem : JoinedOrSeparate<["-"], "stdlib++-isystem">,
  Group<clang_i_Group>,
  HelpText<"Use directory as the C++ standard library include path">,
  Flags<[NoXarchOption]>, MetaVarName<"<directory>">;
def unwindlib_EQ : Joined<["-", "--"], "unwindlib=">, Flags<[CC1Option]>,
  HelpText<"Unwind library to use">, Values<"libgcc,unwindlib,platform">;
def sub__library : JoinedOrSeparate<["-"], "sub_library">;
def sub__umbrella : JoinedOrSeparate<["-"], "sub_umbrella">;
def system_header_prefix : Joined<["--"], "system-header-prefix=">,
  Group<clang_i_Group>, Flags<[CC1Option]>, MetaVarName<"<prefix>">,
  HelpText<"Treat all #include paths starting with <prefix> as including a "
           "system header.">;
def : Separate<["--"], "system-header-prefix">, Alias<system_header_prefix>;
def no_system_header_prefix : Joined<["--"], "no-system-header-prefix=">,
  Group<clang_i_Group>, Flags<[CC1Option]>, MetaVarName<"<prefix>">,
  HelpText<"Treat all #include paths starting with <prefix> as not including a "
           "system header.">;
def : Separate<["--"], "no-system-header-prefix">, Alias<no_system_header_prefix>;
def s : Flag<["-"], "s">, Group<Link_Group>;
def target : Joined<["--"], "target=">, Flags<[NoXarchOption, CoreOption, FlangOption]>,
  HelpText<"Generate code for the given target">;
def darwin_target_variant : Separate<["-"], "darwin-target-variant">,
  Flags<[NoXarchOption, CoreOption]>,
  HelpText<"Generate code for an additional runtime variant of the deployment target">;
def print_supported_cpus : Flag<["-", "--"], "print-supported-cpus">,
  Group<CompileOnly_Group>, Flags<[CC1Option, CoreOption]>,
  HelpText<"Print supported cpu models for the given target (if target is not specified,"
           " it will print the supported cpus for the default target)">,
  MarshallingInfoFlag<FrontendOpts<"PrintSupportedCPUs">>;
def mcpu_EQ_QUESTION : Flag<["-"], "mcpu=?">, Alias<print_supported_cpus>;
def mtune_EQ_QUESTION : Flag<["-"], "mtune=?">, Alias<print_supported_cpus>;
def time : Flag<["-"], "time">,
  HelpText<"Time individual commands">;
def traditional_cpp : Flag<["-", "--"], "traditional-cpp">, Flags<[CC1Option]>,
  HelpText<"Enable some traditional CPP emulation">,
  MarshallingInfoFlag<LangOpts<"TraditionalCPP">>;
def traditional : Flag<["-", "--"], "traditional">;
def trigraphs : Flag<["-", "--"], "trigraphs">, Alias<ftrigraphs>,
  HelpText<"Process trigraph sequences">;
def twolevel__namespace__hints : Flag<["-"], "twolevel_namespace_hints">;
def twolevel__namespace : Flag<["-"], "twolevel_namespace">;
def t : Flag<["-"], "t">, Group<Link_Group>;
def umbrella : Separate<["-"], "umbrella">;
def undefined : JoinedOrSeparate<["-"], "undefined">, Group<u_Group>;
def undef : Flag<["-"], "undef">, Group<u_Group>, Flags<[CC1Option]>,
  HelpText<"undef all system defines">,
  MarshallingInfoNegativeFlag<PreprocessorOpts<"UsePredefines">>;
def unexported__symbols__list : Separate<["-"], "unexported_symbols_list">;
def u : JoinedOrSeparate<["-"], "u">, Group<u_Group>;
def v : Flag<["-"], "v">, Flags<[CC1Option, CoreOption]>,
  HelpText<"Show commands to run and use verbose output">,
  MarshallingInfoFlag<HeaderSearchOpts<"Verbose">>;
def altivec_src_compat : Joined<["-"], "faltivec-src-compat=">,
  Flags<[CC1Option]>, Group<f_Group>,
  HelpText<"Source-level compatibility for Altivec vectors (for PowerPC "
           "targets). This includes results of vector comparison (scalar for "
           "'xl', vector for 'gcc') as well as behavior when initializing with "
           "a scalar (splatting for 'xl', element zero only for 'gcc'). For "
           "'mixed', the compatibility is as 'gcc' for 'vector bool/vector "
           "pixel' and as 'xl' for other types. Current default is 'mixed'.">,
  Values<"mixed,gcc,xl">,
  NormalizedValuesScope<"LangOptions::AltivecSrcCompatKind">,
  NormalizedValues<["Mixed", "GCC", "XL"]>,
  MarshallingInfoEnum<LangOpts<"AltivecSrcCompat">, "Mixed">;
def verify_debug_info : Flag<["--"], "verify-debug-info">, Flags<[NoXarchOption]>,
  HelpText<"Verify the binary representation of debug output">;
def weak_l : Joined<["-"], "weak-l">, Flags<[LinkerInput]>;
def weak__framework : Separate<["-"], "weak_framework">, Flags<[LinkerInput]>;
def weak__library : Separate<["-"], "weak_library">, Flags<[LinkerInput]>;
def weak__reference__mismatches : Separate<["-"], "weak_reference_mismatches">;
def whatsloaded : Flag<["-"], "whatsloaded">;
def why_load : Flag<["-"], "why_load">;
def whyload : Flag<["-"], "whyload">, Alias<why_load>;
def w : Flag<["-"], "w">, HelpText<"Suppress all warnings">, Flags<[CC1Option]>,
  MarshallingInfoFlag<DiagnosticOpts<"IgnoreWarnings">>;
def x : JoinedOrSeparate<["-"], "x">,
Flags<[NoXarchOption,CC1Option,FlangOption,FC1Option]>,
  HelpText<"Treat subsequent input files as having type <language>">,
  MetaVarName<"<language>">;
def y : Joined<["-"], "y">;

defm integrated_as : BoolFOption<"integrated-as",
  CodeGenOpts<"DisableIntegratedAS">, DefaultFalse,
  NegFlag<SetTrue, [CC1Option, FlangOption], "Disable">, PosFlag<SetFalse, [], "Enable">,
  BothFlags<[], " the integrated assembler">>;

def fintegrated_cc1 : Flag<["-"], "fintegrated-cc1">,
                      Flags<[CoreOption, NoXarchOption]>, Group<f_Group>,
                      HelpText<"Run cc1 in-process">;
def fno_integrated_cc1 : Flag<["-"], "fno-integrated-cc1">,
                         Flags<[CoreOption, NoXarchOption]>, Group<f_Group>,
                         HelpText<"Spawn a separate process for each cc1">;

def fintegrated_objemitter : Flag<["-"], "fintegrated-objemitter">,
  Flags<[CoreOption, NoXarchOption]>, Group<f_Group>,
  HelpText<"Use internal machine object code emitter.">;
def fno_integrated_objemitter : Flag<["-"], "fno-integrated-objemitter">,
  Flags<[CoreOption, NoXarchOption]>, Group<f_Group>,
  HelpText<"Use external machine object code emitter.">;

def : Flag<["-"], "integrated-as">, Alias<fintegrated_as>, Flags<[NoXarchOption]>;
def : Flag<["-"], "no-integrated-as">, Alias<fno_integrated_as>,
      Flags<[CC1Option, FlangOption, NoXarchOption]>;

def working_directory : JoinedOrSeparate<["-"], "working-directory">, Flags<[CC1Option]>,
  HelpText<"Resolve file paths relative to the specified directory">,
  MarshallingInfoString<FileSystemOpts<"WorkingDir">>;
def working_directory_EQ : Joined<["-"], "working-directory=">, Flags<[CC1Option]>,
  Alias<working_directory>;

// Double dash options, which are usually an alias for one of the previous
// options.

def _mhwdiv_EQ : Joined<["--"], "mhwdiv=">, Alias<mhwdiv_EQ>;
def _mhwdiv : Separate<["--"], "mhwdiv">, Alias<mhwdiv_EQ>;
def _CLASSPATH_EQ : Joined<["--"], "CLASSPATH=">, Alias<fclasspath_EQ>;
def _CLASSPATH : Separate<["--"], "CLASSPATH">, Alias<fclasspath_EQ>;
def _all_warnings : Flag<["--"], "all-warnings">, Alias<Wall>;
def _analyzer_no_default_checks : Flag<["--"], "analyzer-no-default-checks">, Flags<[NoXarchOption]>;
def _analyzer_output : JoinedOrSeparate<["--"], "analyzer-output">, Flags<[NoXarchOption]>,
  HelpText<"Static analyzer report output format (html|plist|plist-multi-file|plist-html|sarif|sarif-html|text).">;
def _analyze : Flag<["--"], "analyze">, Flags<[NoXarchOption, CoreOption]>,
  HelpText<"Run the static analyzer">;
def _assemble : Flag<["--"], "assemble">, Alias<S>;
def _assert_EQ : Joined<["--"], "assert=">, Alias<A>;
def _assert : Separate<["--"], "assert">, Alias<A>;
def _bootclasspath_EQ : Joined<["--"], "bootclasspath=">, Alias<fbootclasspath_EQ>;
def _bootclasspath : Separate<["--"], "bootclasspath">, Alias<fbootclasspath_EQ>;
def _classpath_EQ : Joined<["--"], "classpath=">, Alias<fclasspath_EQ>;
def _classpath : Separate<["--"], "classpath">, Alias<fclasspath_EQ>;
def _comments_in_macros : Flag<["--"], "comments-in-macros">, Alias<CC>;
def _comments : Flag<["--"], "comments">, Alias<C>;
def _compile : Flag<["--"], "compile">, Alias<c>;
def _constant_cfstrings : Flag<["--"], "constant-cfstrings">;
def _debug_EQ : Joined<["--"], "debug=">, Alias<g_Flag>;
def _debug : Flag<["--"], "debug">, Alias<g_Flag>;
def _define_macro_EQ : Joined<["--"], "define-macro=">, Alias<D>;
def _define_macro : Separate<["--"], "define-macro">, Alias<D>;
def _dependencies : Flag<["--"], "dependencies">, Alias<M>;
def _dyld_prefix_EQ : Joined<["--"], "dyld-prefix=">;
def _dyld_prefix : Separate<["--"], "dyld-prefix">, Alias<_dyld_prefix_EQ>;
def _encoding_EQ : Joined<["--"], "encoding=">, Alias<fencoding_EQ>;
def _encoding : Separate<["--"], "encoding">, Alias<fencoding_EQ>;
def _entry : Flag<["--"], "entry">, Alias<e>;
def _extdirs_EQ : Joined<["--"], "extdirs=">, Alias<fextdirs_EQ>;
def _extdirs : Separate<["--"], "extdirs">, Alias<fextdirs_EQ>;
def _extra_warnings : Flag<["--"], "extra-warnings">, Alias<W_Joined>;
def _for_linker_EQ : Joined<["--"], "for-linker=">, Alias<Xlinker>;
def _for_linker : Separate<["--"], "for-linker">, Alias<Xlinker>;
def _force_link_EQ : Joined<["--"], "force-link=">, Alias<u>;
def _force_link : Separate<["--"], "force-link">, Alias<u>;
def _help_hidden : Flag<["--"], "help-hidden">,
  HelpText<"Display help for hidden options">;
def _imacros_EQ : Joined<["--"], "imacros=">, Alias<imacros>;
def _include_barrier : Flag<["--"], "include-barrier">, Alias<I_>;
def _include_directory_after_EQ : Joined<["--"], "include-directory-after=">, Alias<idirafter>;
def _include_directory_after : Separate<["--"], "include-directory-after">, Alias<idirafter>;
def _include_directory_EQ : Joined<["--"], "include-directory=">, Alias<I>;
def _include_directory : Separate<["--"], "include-directory">, Alias<I>;
def _include_prefix_EQ : Joined<["--"], "include-prefix=">, Alias<iprefix>;
def _include_prefix : Separate<["--"], "include-prefix">, Alias<iprefix>;
def _include_with_prefix_after_EQ : Joined<["--"], "include-with-prefix-after=">, Alias<iwithprefix>;
def _include_with_prefix_after : Separate<["--"], "include-with-prefix-after">, Alias<iwithprefix>;
def _include_with_prefix_before_EQ : Joined<["--"], "include-with-prefix-before=">, Alias<iwithprefixbefore>;
def _include_with_prefix_before : Separate<["--"], "include-with-prefix-before">, Alias<iwithprefixbefore>;
def _include_with_prefix_EQ : Joined<["--"], "include-with-prefix=">, Alias<iwithprefix>;
def _include_with_prefix : Separate<["--"], "include-with-prefix">, Alias<iwithprefix>;
def _include_EQ : Joined<["--"], "include=">, Alias<include_>;
def _language_EQ : Joined<["--"], "language=">, Alias<x>;
def _language : Separate<["--"], "language">, Alias<x>;
def _library_directory_EQ : Joined<["--"], "library-directory=">, Alias<L>;
def _library_directory : Separate<["--"], "library-directory">, Alias<L>;
def _no_line_commands : Flag<["--"], "no-line-commands">, Alias<P>;
def _no_standard_includes : Flag<["--"], "no-standard-includes">, Alias<nostdinc>;
def _no_standard_libraries : Flag<["--"], "no-standard-libraries">, Alias<nostdlib>;
def _no_undefined : Flag<["--"], "no-undefined">, Flags<[LinkerInput]>;
def _no_warnings : Flag<["--"], "no-warnings">, Alias<w>;
def _optimize_EQ : Joined<["--"], "optimize=">, Alias<O>;
def _optimize : Flag<["--"], "optimize">, Alias<O>;
def _output_class_directory_EQ : Joined<["--"], "output-class-directory=">, Alias<foutput_class_dir_EQ>;
def _output_class_directory : Separate<["--"], "output-class-directory">, Alias<foutput_class_dir_EQ>;
def _output_EQ : Joined<["--"], "output=">, Alias<o>;
def _output : Separate<["--"], "output">, Alias<o>;
def _param : Separate<["--"], "param">, Group<CompileOnly_Group>;
def _param_EQ : Joined<["--"], "param=">, Alias<_param>;
def _precompile : Flag<["--"], "precompile">, Flags<[NoXarchOption]>,
  Group<Action_Group>, HelpText<"Only precompile the input">;
def _prefix_EQ : Joined<["--"], "prefix=">, Alias<B>;
def _prefix : Separate<["--"], "prefix">, Alias<B>;
def _preprocess : Flag<["--"], "preprocess">, Alias<E>;
def _print_diagnostic_categories : Flag<["--"], "print-diagnostic-categories">;
def _print_file_name : Separate<["--"], "print-file-name">, Alias<print_file_name_EQ>;
def _print_missing_file_dependencies : Flag<["--"], "print-missing-file-dependencies">, Alias<MG>;
def _print_prog_name : Separate<["--"], "print-prog-name">, Alias<print_prog_name_EQ>;
def _profile_blocks : Flag<["--"], "profile-blocks">, Alias<a>;
def _profile : Flag<["--"], "profile">, Alias<p>;
def _resource_EQ : Joined<["--"], "resource=">, Alias<fcompile_resource_EQ>;
def _resource : Separate<["--"], "resource">, Alias<fcompile_resource_EQ>;
def _rtlib : Separate<["--"], "rtlib">, Alias<rtlib_EQ>;
def _serialize_diags : Separate<["-", "--"], "serialize-diagnostics">, Flags<[NoXarchOption]>,
  HelpText<"Serialize compiler diagnostics to a file">;
// We give --version different semantics from -version.
def _version : Flag<["--"], "version">,
  Flags<[CoreOption, FlangOption]>,
  HelpText<"Print version information">;
def _signed_char : Flag<["--"], "signed-char">, Alias<fsigned_char>;
def _std : Separate<["--"], "std">, Alias<std_EQ>;
def _stdlib : Separate<["--"], "stdlib">, Alias<stdlib_EQ>;
def _sysroot_EQ : Joined<["--"], "sysroot=">;
def _sysroot : Separate<["--"], "sysroot">, Alias<_sysroot_EQ>;
def _target_help : Flag<["--"], "target-help">;
def _trace_includes : Flag<["--"], "trace-includes">, Alias<H>;
def _undefine_macro_EQ : Joined<["--"], "undefine-macro=">, Alias<U>;
def _undefine_macro : Separate<["--"], "undefine-macro">, Alias<U>;
def _unsigned_char : Flag<["--"], "unsigned-char">, Alias<funsigned_char>;
def _user_dependencies : Flag<["--"], "user-dependencies">, Alias<MM>;
def _verbose : Flag<["--"], "verbose">, Alias<v>;
def _warn__EQ : Joined<["--"], "warn-=">, Alias<W_Joined>;
def _warn_ : Joined<["--"], "warn-">, Alias<W_Joined>;
def _write_dependencies : Flag<["--"], "write-dependencies">, Alias<MD>;
def _write_user_dependencies : Flag<["--"], "write-user-dependencies">, Alias<MMD>;
def _ : Joined<["--"], "">, Flags<[Unsupported]>;

// Hexagon feature flags.
def mieee_rnd_near : Flag<["-"], "mieee-rnd-near">,
  Group<m_hexagon_Features_Group>;
def mv5 : Flag<["-"], "mv5">, Group<m_hexagon_Features_Group>, Alias<mcpu_EQ>,
  AliasArgs<["hexagonv5"]>;
def mv55 : Flag<["-"], "mv55">, Group<m_hexagon_Features_Group>,
  Alias<mcpu_EQ>, AliasArgs<["hexagonv55"]>;
def mv60 : Flag<["-"], "mv60">, Group<m_hexagon_Features_Group>,
  Alias<mcpu_EQ>, AliasArgs<["hexagonv60"]>;
def mv62 : Flag<["-"], "mv62">, Group<m_hexagon_Features_Group>,
  Alias<mcpu_EQ>, AliasArgs<["hexagonv62"]>;
def mv65 : Flag<["-"], "mv65">, Group<m_hexagon_Features_Group>,
  Alias<mcpu_EQ>, AliasArgs<["hexagonv65"]>;
def mv66 : Flag<["-"], "mv66">, Group<m_hexagon_Features_Group>,
  Alias<mcpu_EQ>, AliasArgs<["hexagonv66"]>;
def mv67 : Flag<["-"], "mv67">, Group<m_hexagon_Features_Group>,
  Alias<mcpu_EQ>, AliasArgs<["hexagonv67"]>;
def mv67t : Flag<["-"], "mv67t">, Group<m_hexagon_Features_Group>,
  Alias<mcpu_EQ>, AliasArgs<["hexagonv67t"]>;
def mv68 : Flag<["-"], "mv68">, Group<m_hexagon_Features_Group>,
  Alias<mcpu_EQ>, AliasArgs<["hexagonv68"]>;
def mv69 : Flag<["-"], "mv69">, Group<m_hexagon_Features_Group>,
  Alias<mcpu_EQ>, AliasArgs<["hexagonv69"]>;
def mhexagon_hvx : Flag<["-"], "mhvx">, Group<m_hexagon_Features_HVX_Group>,
  HelpText<"Enable Hexagon Vector eXtensions">;
def mhexagon_hvx_EQ : Joined<["-"], "mhvx=">,
  Group<m_hexagon_Features_HVX_Group>,
  HelpText<"Enable Hexagon Vector eXtensions">;
def mno_hexagon_hvx : Flag<["-"], "mno-hvx">,
  Group<m_hexagon_Features_HVX_Group>,
  HelpText<"Disable Hexagon Vector eXtensions">;
def mhexagon_hvx_length_EQ : Joined<["-"], "mhvx-length=">,
  Group<m_hexagon_Features_HVX_Group>, HelpText<"Set Hexagon Vector Length">,
  Values<"64B,128B">;
def mhexagon_hvx_qfloat : Flag<["-"], "mhvx-qfloat">,
  Group<m_hexagon_Features_HVX_Group>,
  HelpText<"Enable Hexagon HVX QFloat instructions">;
def mno_hexagon_hvx_qfloat : Flag<["-"], "mno-hvx-qfloat">,
  Group<m_hexagon_Features_HVX_Group>,
  HelpText<"Disable Hexagon HVX QFloat instructions">;
def mhexagon_hvx_ieee_fp : Flag<["-"], "mhvx-ieee-fp">,
  Group<m_hexagon_Features_Group>,
  HelpText<"Enable Hexagon HVX IEEE floating-point">;
def mno_hexagon_hvx_ieee_fp : Flag<["-"], "mno-hvx-ieee-fp">,
  Group<m_hexagon_Features_Group>,
  HelpText<"Disable Hexagon HVX IEEE floating-point">;
def ffixed_r19: Flag<["-"], "ffixed-r19">,
  HelpText<"Reserve register r19 (Hexagon only)">;
def mmemops : Flag<["-"], "mmemops">, Group<m_hexagon_Features_Group>,
  Flags<[CC1Option]>, HelpText<"Enable generation of memop instructions">;
def mno_memops : Flag<["-"], "mno-memops">, Group<m_hexagon_Features_Group>,
  Flags<[CC1Option]>, HelpText<"Disable generation of memop instructions">;
def mpackets : Flag<["-"], "mpackets">, Group<m_hexagon_Features_Group>,
  Flags<[CC1Option]>, HelpText<"Enable generation of instruction packets">;
def mno_packets : Flag<["-"], "mno-packets">, Group<m_hexagon_Features_Group>,
  Flags<[CC1Option]>, HelpText<"Disable generation of instruction packets">;
def mnvj : Flag<["-"], "mnvj">, Group<m_hexagon_Features_Group>,
  Flags<[CC1Option]>, HelpText<"Enable generation of new-value jumps">;
def mno_nvj : Flag<["-"], "mno-nvj">, Group<m_hexagon_Features_Group>,
  Flags<[CC1Option]>, HelpText<"Disable generation of new-value jumps">;
def mnvs : Flag<["-"], "mnvs">, Group<m_hexagon_Features_Group>,
  Flags<[CC1Option]>, HelpText<"Enable generation of new-value stores">;
def mno_nvs : Flag<["-"], "mno-nvs">, Group<m_hexagon_Features_Group>,
  Flags<[CC1Option]>, HelpText<"Disable generation of new-value stores">;

// M68k features flags
def m68000 : Flag<["-"], "m68000">, Group<m_m68k_Features_Group>;
def m68010 : Flag<["-"], "m68010">, Group<m_m68k_Features_Group>;
def m68020 : Flag<["-"], "m68020">, Group<m_m68k_Features_Group>;
def m68030 : Flag<["-"], "m68030">, Group<m_m68k_Features_Group>;
def m68040 : Flag<["-"], "m68040">, Group<m_m68k_Features_Group>;
def m68060 : Flag<["-"], "m68060">, Group<m_m68k_Features_Group>;

foreach i = {0-6} in
  def ffixed_a#i : Flag<["-"], "ffixed-a"#i>, Group<m_m68k_Features_Group>,
    HelpText<"Reserve the a"#i#" register (M68k only)">;
foreach i = {0-7} in
  def ffixed_d#i : Flag<["-"], "ffixed-d"#i>, Group<m_m68k_Features_Group>,
    HelpText<"Reserve the d"#i#" register (M68k only)">;

// X86 feature flags
def mx87 : Flag<["-"], "mx87">, Group<m_x86_Features_Group>;
def mno_x87 : Flag<["-"], "mno-x87">, Group<m_x86_Features_Group>;
def m80387 : Flag<["-"], "m80387">, Alias<mx87>;
def mno_80387 : Flag<["-"], "mno-80387">, Alias<mno_x87>;
def mno_fp_ret_in_387 : Flag<["-"], "mno-fp-ret-in-387">, Alias<mno_x87>;
def mmmx : Flag<["-"], "mmmx">, Group<m_x86_Features_Group>;
def mno_mmx : Flag<["-"], "mno-mmx">, Group<m_x86_Features_Group>;
def m3dnow : Flag<["-"], "m3dnow">, Group<m_x86_Features_Group>;
def mno_3dnow : Flag<["-"], "mno-3dnow">, Group<m_x86_Features_Group>;
def m3dnowa : Flag<["-"], "m3dnowa">, Group<m_x86_Features_Group>;
def mno_3dnowa : Flag<["-"], "mno-3dnowa">, Group<m_x86_Features_Group>;
def mamx_bf16 : Flag<["-"], "mamx-bf16">, Group<m_x86_Features_Group>;
def mno_amx_bf16 : Flag<["-"], "mno-amx-bf16">, Group<m_x86_Features_Group>;
def mtamx_int8 : Flag<["-"], "mamx-int8">, Group<m_x86_Features_Group>;
def mno_amx_int8 : Flag<["-"], "mno-amx-int8">, Group<m_x86_Features_Group>;
def mamx_tile : Flag<["-"], "mamx-tile">, Group<m_x86_Features_Group>;
def mno_amx_tile : Flag<["-"], "mno-amx-tile">, Group<m_x86_Features_Group>;
def msse : Flag<["-"], "msse">, Group<m_x86_Features_Group>;
def mno_sse : Flag<["-"], "mno-sse">, Group<m_x86_Features_Group>;
def msse2 : Flag<["-"], "msse2">, Group<m_x86_Features_Group>;
def mno_sse2 : Flag<["-"], "mno-sse2">, Group<m_x86_Features_Group>;
def msse3 : Flag<["-"], "msse3">, Group<m_x86_Features_Group>;
def mno_sse3 : Flag<["-"], "mno-sse3">, Group<m_x86_Features_Group>;
def mssse3 : Flag<["-"], "mssse3">, Group<m_x86_Features_Group>;
def mno_ssse3 : Flag<["-"], "mno-ssse3">, Group<m_x86_Features_Group>;
def msse4_1 : Flag<["-"], "msse4.1">, Group<m_x86_Features_Group>;
def mno_sse4_1 : Flag<["-"], "mno-sse4.1">, Group<m_x86_Features_Group>;
def msse4_2 : Flag<["-"], "msse4.2">, Group<m_x86_Features_Group>;
def mno_sse4_2 : Flag<["-"], "mno-sse4.2">, Group<m_x86_Features_Group>;
def msse4 : Flag<["-"], "msse4">, Alias<msse4_2>;
// -mno-sse4 turns off sse4.1 which has the effect of turning off everything
// later than 4.1. -msse4 turns on 4.2 which has the effect of turning on
// everything earlier than 4.2.
def mno_sse4 : Flag<["-"], "mno-sse4">, Alias<mno_sse4_1>;
def msse4a : Flag<["-"], "msse4a">, Group<m_x86_Features_Group>;
def mno_sse4a : Flag<["-"], "mno-sse4a">, Group<m_x86_Features_Group>;
def mavx : Flag<["-"], "mavx">, Group<m_x86_Features_Group>;
def mno_avx : Flag<["-"], "mno-avx">, Group<m_x86_Features_Group>;
def mavx2 : Flag<["-"], "mavx2">, Group<m_x86_Features_Group>;
def mno_avx2 : Flag<["-"], "mno-avx2">, Group<m_x86_Features_Group>;
def mavx512f : Flag<["-"], "mavx512f">, Group<m_x86_Features_Group>;
def mno_avx512f : Flag<["-"], "mno-avx512f">, Group<m_x86_Features_Group>;
def mavx512bf16 : Flag<["-"], "mavx512bf16">, Group<m_x86_Features_Group>;
def mno_avx512bf16 : Flag<["-"], "mno-avx512bf16">, Group<m_x86_Features_Group>;
def mavx512bitalg : Flag<["-"], "mavx512bitalg">, Group<m_x86_Features_Group>;
def mno_avx512bitalg : Flag<["-"], "mno-avx512bitalg">, Group<m_x86_Features_Group>;
def mavx512bw : Flag<["-"], "mavx512bw">, Group<m_x86_Features_Group>;
def mno_avx512bw : Flag<["-"], "mno-avx512bw">, Group<m_x86_Features_Group>;
def mavx512cd : Flag<["-"], "mavx512cd">, Group<m_x86_Features_Group>;
def mno_avx512cd : Flag<["-"], "mno-avx512cd">, Group<m_x86_Features_Group>;
def mavx512dq : Flag<["-"], "mavx512dq">, Group<m_x86_Features_Group>;
def mno_avx512dq : Flag<["-"], "mno-avx512dq">, Group<m_x86_Features_Group>;
def mavx512er : Flag<["-"], "mavx512er">, Group<m_x86_Features_Group>;
def mno_avx512er : Flag<["-"], "mno-avx512er">, Group<m_x86_Features_Group>;
def mavx512fp16 : Flag<["-"], "mavx512fp16">, Group<m_x86_Features_Group>;
def mno_avx512fp16 : Flag<["-"], "mno-avx512fp16">, Group<m_x86_Features_Group>;
def mavx512ifma : Flag<["-"], "mavx512ifma">, Group<m_x86_Features_Group>;
def mno_avx512ifma : Flag<["-"], "mno-avx512ifma">, Group<m_x86_Features_Group>;
def mavx512pf : Flag<["-"], "mavx512pf">, Group<m_x86_Features_Group>;
def mno_avx512pf : Flag<["-"], "mno-avx512pf">, Group<m_x86_Features_Group>;
def mavx512vbmi : Flag<["-"], "mavx512vbmi">, Group<m_x86_Features_Group>;
def mno_avx512vbmi : Flag<["-"], "mno-avx512vbmi">, Group<m_x86_Features_Group>;
def mavx512vbmi2 : Flag<["-"], "mavx512vbmi2">, Group<m_x86_Features_Group>;
def mno_avx512vbmi2 : Flag<["-"], "mno-avx512vbmi2">, Group<m_x86_Features_Group>;
def mavx512vl : Flag<["-"], "mavx512vl">, Group<m_x86_Features_Group>;
def mno_avx512vl : Flag<["-"], "mno-avx512vl">, Group<m_x86_Features_Group>;
def mavx512vnni : Flag<["-"], "mavx512vnni">, Group<m_x86_Features_Group>;
def mno_avx512vnni : Flag<["-"], "mno-avx512vnni">, Group<m_x86_Features_Group>;
def mavx512vpopcntdq : Flag<["-"], "mavx512vpopcntdq">, Group<m_x86_Features_Group>;
def mno_avx512vpopcntdq : Flag<["-"], "mno-avx512vpopcntdq">, Group<m_x86_Features_Group>;
def mavx512vp2intersect : Flag<["-"], "mavx512vp2intersect">, Group<m_x86_Features_Group>;
def mno_avx512vp2intersect : Flag<["-"], "mno-avx512vp2intersect">, Group<m_x86_Features_Group>;
def mavxvnni : Flag<["-"], "mavxvnni">, Group<m_x86_Features_Group>;
def mno_avxvnni : Flag<["-"], "mno-avxvnni">, Group<m_x86_Features_Group>;
def madx : Flag<["-"], "madx">, Group<m_x86_Features_Group>;
def mno_adx : Flag<["-"], "mno-adx">, Group<m_x86_Features_Group>;
def maes : Flag<["-"], "maes">, Group<m_x86_Features_Group>;
def mno_aes : Flag<["-"], "mno-aes">, Group<m_x86_Features_Group>;
def mbmi : Flag<["-"], "mbmi">, Group<m_x86_Features_Group>;
def mno_bmi : Flag<["-"], "mno-bmi">, Group<m_x86_Features_Group>;
def mbmi2 : Flag<["-"], "mbmi2">, Group<m_x86_Features_Group>;
def mno_bmi2 : Flag<["-"], "mno-bmi2">, Group<m_x86_Features_Group>;
def mcldemote : Flag<["-"], "mcldemote">, Group<m_x86_Features_Group>;
def mno_cldemote : Flag<["-"], "mno-cldemote">, Group<m_x86_Features_Group>;
def mclflushopt : Flag<["-"], "mclflushopt">, Group<m_x86_Features_Group>;
def mno_clflushopt : Flag<["-"], "mno-clflushopt">, Group<m_x86_Features_Group>;
def mclwb : Flag<["-"], "mclwb">, Group<m_x86_Features_Group>;
def mno_clwb : Flag<["-"], "mno-clwb">, Group<m_x86_Features_Group>;
def mwbnoinvd : Flag<["-"], "mwbnoinvd">, Group<m_x86_Features_Group>;
def mno_wbnoinvd : Flag<["-"], "mno-wbnoinvd">, Group<m_x86_Features_Group>;
def mclzero : Flag<["-"], "mclzero">, Group<m_x86_Features_Group>;
def mno_clzero : Flag<["-"], "mno-clzero">, Group<m_x86_Features_Group>;
def mcrc32 : Flag<["-"], "mcrc32">, Group<m_x86_Features_Group>;
def mno_crc32 : Flag<["-"], "mno-crc32">, Group<m_x86_Features_Group>;
def mcx16 : Flag<["-"], "mcx16">, Group<m_x86_Features_Group>;
def mno_cx16 : Flag<["-"], "mno-cx16">, Group<m_x86_Features_Group>;
def menqcmd : Flag<["-"], "menqcmd">, Group<m_x86_Features_Group>;
def mno_enqcmd : Flag<["-"], "mno-enqcmd">, Group<m_x86_Features_Group>;
def mf16c : Flag<["-"], "mf16c">, Group<m_x86_Features_Group>;
def mno_f16c : Flag<["-"], "mno-f16c">, Group<m_x86_Features_Group>;
def mfma : Flag<["-"], "mfma">, Group<m_x86_Features_Group>;
def mno_fma : Flag<["-"], "mno-fma">, Group<m_x86_Features_Group>;
def mfma4 : Flag<["-"], "mfma4">, Group<m_x86_Features_Group>;
def mno_fma4 : Flag<["-"], "mno-fma4">, Group<m_x86_Features_Group>;
def mfsgsbase : Flag<["-"], "mfsgsbase">, Group<m_x86_Features_Group>;
def mno_fsgsbase : Flag<["-"], "mno-fsgsbase">, Group<m_x86_Features_Group>;
def mfxsr : Flag<["-"], "mfxsr">, Group<m_x86_Features_Group>;
def mno_fxsr : Flag<["-"], "mno-fxsr">, Group<m_x86_Features_Group>;
def minvpcid : Flag<["-"], "minvpcid">, Group<m_x86_Features_Group>;
def mno_invpcid : Flag<["-"], "mno-invpcid">, Group<m_x86_Features_Group>;
def mgfni : Flag<["-"], "mgfni">, Group<m_x86_Features_Group>;
def mno_gfni : Flag<["-"], "mno-gfni">, Group<m_x86_Features_Group>;
def mhreset : Flag<["-"], "mhreset">, Group<m_x86_Features_Group>;
def mno_hreset : Flag<["-"], "mno-hreset">, Group<m_x86_Features_Group>;
def mkl : Flag<["-"], "mkl">, Group<m_x86_Features_Group>;
def mno_kl : Flag<["-"], "mno-kl">, Group<m_x86_Features_Group>;
def mwidekl : Flag<["-"], "mwidekl">, Group<m_x86_Features_Group>;
def mno_widekl : Flag<["-"], "mno-widekl">, Group<m_x86_Features_Group>;
def mlwp : Flag<["-"], "mlwp">, Group<m_x86_Features_Group>;
def mno_lwp : Flag<["-"], "mno-lwp">, Group<m_x86_Features_Group>;
def mlzcnt : Flag<["-"], "mlzcnt">, Group<m_x86_Features_Group>;
def mno_lzcnt : Flag<["-"], "mno-lzcnt">, Group<m_x86_Features_Group>;
def mmovbe : Flag<["-"], "mmovbe">, Group<m_x86_Features_Group>;
def mno_movbe : Flag<["-"], "mno-movbe">, Group<m_x86_Features_Group>;
def mmovdiri : Flag<["-"], "mmovdiri">, Group<m_x86_Features_Group>;
def mno_movdiri : Flag<["-"], "mno-movdiri">, Group<m_x86_Features_Group>;
def mmovdir64b : Flag<["-"], "mmovdir64b">, Group<m_x86_Features_Group>;
def mno_movdir64b : Flag<["-"], "mno-movdir64b">, Group<m_x86_Features_Group>;
def mmwaitx : Flag<["-"], "mmwaitx">, Group<m_x86_Features_Group>;
def mno_mwaitx : Flag<["-"], "mno-mwaitx">, Group<m_x86_Features_Group>;
def mpku : Flag<["-"], "mpku">, Group<m_x86_Features_Group>;
def mno_pku : Flag<["-"], "mno-pku">, Group<m_x86_Features_Group>;
def mpclmul : Flag<["-"], "mpclmul">, Group<m_x86_Features_Group>;
def mno_pclmul : Flag<["-"], "mno-pclmul">, Group<m_x86_Features_Group>;
def mpconfig : Flag<["-"], "mpconfig">, Group<m_x86_Features_Group>;
def mno_pconfig : Flag<["-"], "mno-pconfig">, Group<m_x86_Features_Group>;
def mpopcnt : Flag<["-"], "mpopcnt">, Group<m_x86_Features_Group>;
def mno_popcnt : Flag<["-"], "mno-popcnt">, Group<m_x86_Features_Group>;
def mprefetchwt1 : Flag<["-"], "mprefetchwt1">, Group<m_x86_Features_Group>;
def mno_prefetchwt1 : Flag<["-"], "mno-prefetchwt1">, Group<m_x86_Features_Group>;
def mprfchw : Flag<["-"], "mprfchw">, Group<m_x86_Features_Group>;
def mno_prfchw : Flag<["-"], "mno-prfchw">, Group<m_x86_Features_Group>;
def mptwrite : Flag<["-"], "mptwrite">, Group<m_x86_Features_Group>;
def mno_ptwrite : Flag<["-"], "mno-ptwrite">, Group<m_x86_Features_Group>;
def mrdpid : Flag<["-"], "mrdpid">, Group<m_x86_Features_Group>;
def mno_rdpid : Flag<["-"], "mno-rdpid">, Group<m_x86_Features_Group>;
def mrdpru : Flag<["-"], "mrdpru">, Group<m_x86_Features_Group>;
def mno_rdpru : Flag<["-"], "mno-rdpru">, Group<m_x86_Features_Group>;
def mrdrnd : Flag<["-"], "mrdrnd">, Group<m_x86_Features_Group>;
def mno_rdrnd : Flag<["-"], "mno-rdrnd">, Group<m_x86_Features_Group>;
def mrtm : Flag<["-"], "mrtm">, Group<m_x86_Features_Group>;
def mno_rtm : Flag<["-"], "mno-rtm">, Group<m_x86_Features_Group>;
def mrdseed : Flag<["-"], "mrdseed">, Group<m_x86_Features_Group>;
def mno_rdseed : Flag<["-"], "mno-rdseed">, Group<m_x86_Features_Group>;
def msahf : Flag<["-"], "msahf">, Group<m_x86_Features_Group>;
def mno_sahf : Flag<["-"], "mno-sahf">, Group<m_x86_Features_Group>;
def mserialize : Flag<["-"], "mserialize">, Group<m_x86_Features_Group>;
def mno_serialize : Flag<["-"], "mno-serialize">, Group<m_x86_Features_Group>;
def msgx : Flag<["-"], "msgx">, Group<m_x86_Features_Group>;
def mno_sgx : Flag<["-"], "mno-sgx">, Group<m_x86_Features_Group>;
def msha : Flag<["-"], "msha">, Group<m_x86_Features_Group>;
def mno_sha : Flag<["-"], "mno-sha">, Group<m_x86_Features_Group>;
def mtbm : Flag<["-"], "mtbm">, Group<m_x86_Features_Group>;
def mno_tbm : Flag<["-"], "mno-tbm">, Group<m_x86_Features_Group>;
def mtsxldtrk : Flag<["-"], "mtsxldtrk">, Group<m_x86_Features_Group>;
def mno_tsxldtrk : Flag<["-"], "mno-tsxldtrk">, Group<m_x86_Features_Group>;
def muintr : Flag<["-"], "muintr">, Group<m_x86_Features_Group>;
def mno_uintr : Flag<["-"], "mno-uintr">, Group<m_x86_Features_Group>;
def mvaes : Flag<["-"], "mvaes">, Group<m_x86_Features_Group>;
def mno_vaes : Flag<["-"], "mno-vaes">, Group<m_x86_Features_Group>;
def mvpclmulqdq : Flag<["-"], "mvpclmulqdq">, Group<m_x86_Features_Group>;
def mno_vpclmulqdq : Flag<["-"], "mno-vpclmulqdq">, Group<m_x86_Features_Group>;
def mwaitpkg : Flag<["-"], "mwaitpkg">, Group<m_x86_Features_Group>;
def mno_waitpkg : Flag<["-"], "mno-waitpkg">, Group<m_x86_Features_Group>;
def mxop : Flag<["-"], "mxop">, Group<m_x86_Features_Group>;
def mno_xop : Flag<["-"], "mno-xop">, Group<m_x86_Features_Group>;
def mxsave : Flag<["-"], "mxsave">, Group<m_x86_Features_Group>;
def mno_xsave : Flag<["-"], "mno-xsave">, Group<m_x86_Features_Group>;
def mxsavec : Flag<["-"], "mxsavec">, Group<m_x86_Features_Group>;
def mno_xsavec : Flag<["-"], "mno-xsavec">, Group<m_x86_Features_Group>;
def mxsaveopt : Flag<["-"], "mxsaveopt">, Group<m_x86_Features_Group>;
def mno_xsaveopt : Flag<["-"], "mno-xsaveopt">, Group<m_x86_Features_Group>;
def mxsaves : Flag<["-"], "mxsaves">, Group<m_x86_Features_Group>;
def mno_xsaves : Flag<["-"], "mno-xsaves">, Group<m_x86_Features_Group>;
def mshstk : Flag<["-"], "mshstk">, Group<m_x86_Features_Group>;
def mno_shstk : Flag<["-"], "mno-shstk">, Group<m_x86_Features_Group>;
def mretpoline_external_thunk : Flag<["-"], "mretpoline-external-thunk">, Group<m_x86_Features_Group>;
def mno_retpoline_external_thunk : Flag<["-"], "mno-retpoline-external-thunk">, Group<m_x86_Features_Group>;
def mvzeroupper : Flag<["-"], "mvzeroupper">, Group<m_x86_Features_Group>;
def mno_vzeroupper : Flag<["-"], "mno-vzeroupper">, Group<m_x86_Features_Group>;

// These are legacy user-facing driver-level option spellings. They are always
// aliases for options that are spelled using the more common Unix / GNU flag
// style of double-dash and equals-joined flags.
def target_legacy_spelling : Separate<["-"], "target">,
                             Alias<target>,
                             Flags<[CoreOption]>;

// Special internal option to handle -Xlinker --no-demangle.
def Z_Xlinker__no_demangle : Flag<["-"], "Z-Xlinker-no-demangle">,
    Flags<[Unsupported, NoArgumentUnused]>;

// Special internal option to allow forwarding arbitrary arguments to linker.
def Zlinker_input : Separate<["-"], "Zlinker-input">,
    Flags<[Unsupported, NoArgumentUnused]>;

// Reserved library options.
def Z_reserved_lib_stdcxx : Flag<["-"], "Z-reserved-lib-stdc++">,
    Flags<[LinkerInput, NoArgumentUnused, Unsupported]>, Group<reserved_lib_Group>;
def Z_reserved_lib_cckext : Flag<["-"], "Z-reserved-lib-cckext">,
    Flags<[LinkerInput, NoArgumentUnused, Unsupported]>, Group<reserved_lib_Group>;

// Ignored options
multiclass BooleanFFlag<string name> {
  def f#NAME : Flag<["-"], "f"#name>;
  def fno_#NAME : Flag<["-"], "fno-"#name>;
}

defm : BooleanFFlag<"keep-inline-functions">, Group<clang_ignored_gcc_optimization_f_Group>;

def fprofile_dir : Joined<["-"], "fprofile-dir=">, Group<f_Group>;

// The default value matches BinutilsVersion in MCAsmInfo.h.
def fbinutils_version_EQ : Joined<["-"], "fbinutils-version=">,
  MetaVarName<"<major.minor>">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Produced object files can use all ELF features supported by this "
  "binutils version and newer. If -fno-integrated-as is specified, the "
  "generated assembly will consider GNU as support. 'none' means that all ELF "
  "features can be used, regardless of binutils support. Defaults to 2.26.">;
def fuse_ld_EQ : Joined<["-"], "fuse-ld=">, Group<f_Group>, Flags<[CoreOption, LinkOption]>;
def ld_path_EQ : Joined<["--"], "ld-path=">, Group<Link_Group>;

defm align_labels : BooleanFFlag<"align-labels">, Group<clang_ignored_gcc_optimization_f_Group>;
def falign_labels_EQ : Joined<["-"], "falign-labels=">, Group<clang_ignored_gcc_optimization_f_Group>;
defm align_loops : BooleanFFlag<"align-loops">, Group<clang_ignored_gcc_optimization_f_Group>;
defm align_jumps : BooleanFFlag<"align-jumps">, Group<clang_ignored_gcc_optimization_f_Group>;
def falign_jumps_EQ : Joined<["-"], "falign-jumps=">, Group<clang_ignored_gcc_optimization_f_Group>;

// FIXME: This option should be supported and wired up to our diognostics, but
// ignore it for now to avoid breaking builds that use it.
def fdiagnostics_show_location_EQ : Joined<["-"], "fdiagnostics-show-location=">, Group<clang_ignored_f_Group>;

defm fcheck_new : BooleanFFlag<"check-new">, Group<clang_ignored_f_Group>;
defm caller_saves : BooleanFFlag<"caller-saves">, Group<clang_ignored_gcc_optimization_f_Group>;
defm reorder_blocks : BooleanFFlag<"reorder-blocks">, Group<clang_ignored_gcc_optimization_f_Group>;
defm branch_count_reg : BooleanFFlag<"branch-count-reg">, Group<clang_ignored_gcc_optimization_f_Group>;
defm default_inline : BooleanFFlag<"default-inline">, Group<clang_ignored_gcc_optimization_f_Group>;
defm fat_lto_objects : BooleanFFlag<"fat-lto-objects">, Group<clang_ignored_gcc_optimization_f_Group>;
defm float_store : BooleanFFlag<"float-store">, Group<clang_ignored_gcc_optimization_f_Group>;
defm friend_injection : BooleanFFlag<"friend-injection">, Group<clang_ignored_f_Group>;
defm function_attribute_list : BooleanFFlag<"function-attribute-list">, Group<clang_ignored_f_Group>;
defm gcse : BooleanFFlag<"gcse">, Group<clang_ignored_gcc_optimization_f_Group>;
defm gcse_after_reload: BooleanFFlag<"gcse-after-reload">, Group<clang_ignored_gcc_optimization_f_Group>;
defm gcse_las: BooleanFFlag<"gcse-las">, Group<clang_ignored_gcc_optimization_f_Group>;
defm gcse_sm: BooleanFFlag<"gcse-sm">, Group<clang_ignored_gcc_optimization_f_Group>;
defm gnu : BooleanFFlag<"gnu">, Group<clang_ignored_f_Group>;
defm implicit_templates : BooleanFFlag<"implicit-templates">, Group<clang_ignored_f_Group>;
defm implement_inlines : BooleanFFlag<"implement-inlines">, Group<clang_ignored_f_Group>;
defm merge_constants : BooleanFFlag<"merge-constants">, Group<clang_ignored_gcc_optimization_f_Group>;
defm modulo_sched : BooleanFFlag<"modulo-sched">, Group<clang_ignored_gcc_optimization_f_Group>;
defm modulo_sched_allow_regmoves : BooleanFFlag<"modulo-sched-allow-regmoves">,
    Group<clang_ignored_gcc_optimization_f_Group>;
defm inline_functions_called_once : BooleanFFlag<"inline-functions-called-once">,
    Group<clang_ignored_gcc_optimization_f_Group>;
def finline_limit_EQ : Joined<["-"], "finline-limit=">, Group<clang_ignored_gcc_optimization_f_Group>;
defm finline_limit : BooleanFFlag<"inline-limit">, Group<clang_ignored_gcc_optimization_f_Group>;
defm inline_small_functions : BooleanFFlag<"inline-small-functions">,
    Group<clang_ignored_gcc_optimization_f_Group>;
defm ipa_cp : BooleanFFlag<"ipa-cp">,
    Group<clang_ignored_gcc_optimization_f_Group>;
defm ivopts : BooleanFFlag<"ivopts">, Group<clang_ignored_gcc_optimization_f_Group>;
defm semantic_interposition : BoolFOption<"semantic-interposition",
  LangOpts<"SemanticInterposition">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option]>, NegFlag<SetFalse>>;
defm non_call_exceptions : BooleanFFlag<"non-call-exceptions">, Group<clang_ignored_f_Group>;
defm peel_loops : BooleanFFlag<"peel-loops">, Group<clang_ignored_gcc_optimization_f_Group>;
defm permissive : BooleanFFlag<"permissive">, Group<clang_ignored_f_Group>;
defm prefetch_loop_arrays : BooleanFFlag<"prefetch-loop-arrays">, Group<clang_ignored_gcc_optimization_f_Group>;
defm printf : BooleanFFlag<"printf">, Group<clang_ignored_f_Group>;
defm profile : BooleanFFlag<"profile">, Group<clang_ignored_f_Group>;
defm profile_correction : BooleanFFlag<"profile-correction">, Group<clang_ignored_gcc_optimization_f_Group>;
defm profile_generate_sampling : BooleanFFlag<"profile-generate-sampling">, Group<clang_ignored_f_Group>;
defm profile_reusedist : BooleanFFlag<"profile-reusedist">, Group<clang_ignored_f_Group>;
defm profile_values : BooleanFFlag<"profile-values">, Group<clang_ignored_gcc_optimization_f_Group>;
defm regs_graph : BooleanFFlag<"regs-graph">, Group<clang_ignored_f_Group>;
defm rename_registers : BooleanFFlag<"rename-registers">, Group<clang_ignored_gcc_optimization_f_Group>;
defm ripa : BooleanFFlag<"ripa">, Group<clang_ignored_f_Group>;
defm schedule_insns : BooleanFFlag<"schedule-insns">, Group<clang_ignored_gcc_optimization_f_Group>;
defm schedule_insns2 : BooleanFFlag<"schedule-insns2">, Group<clang_ignored_gcc_optimization_f_Group>;
defm see : BooleanFFlag<"see">, Group<clang_ignored_f_Group>;
defm signaling_nans : BooleanFFlag<"signaling-nans">, Group<clang_ignored_gcc_optimization_f_Group>;
defm single_precision_constant : BooleanFFlag<"single-precision-constant">,
    Group<clang_ignored_gcc_optimization_f_Group>;
defm spec_constr_count : BooleanFFlag<"spec-constr-count">, Group<clang_ignored_f_Group>;
defm stack_check : BooleanFFlag<"stack-check">, Group<clang_ignored_f_Group>;
defm strength_reduce :
    BooleanFFlag<"strength-reduce">, Group<clang_ignored_gcc_optimization_f_Group>;
defm tls_model : BooleanFFlag<"tls-model">, Group<clang_ignored_f_Group>;
defm tracer : BooleanFFlag<"tracer">, Group<clang_ignored_gcc_optimization_f_Group>;
defm tree_dce : BooleanFFlag<"tree-dce">, Group<clang_ignored_gcc_optimization_f_Group>;
defm tree_salias : BooleanFFlag<"tree-salias">, Group<clang_ignored_f_Group>;
defm tree_ter : BooleanFFlag<"tree-ter">, Group<clang_ignored_gcc_optimization_f_Group>;
defm tree_vectorizer_verbose : BooleanFFlag<"tree-vectorizer-verbose">, Group<clang_ignored_f_Group>;
defm tree_vrp : BooleanFFlag<"tree-vrp">, Group<clang_ignored_gcc_optimization_f_Group>;
defm : BooleanFFlag<"unit-at-a-time">, Group<clang_ignored_gcc_optimization_f_Group>;
defm unroll_all_loops : BooleanFFlag<"unroll-all-loops">, Group<clang_ignored_gcc_optimization_f_Group>;
defm unsafe_loop_optimizations : BooleanFFlag<"unsafe-loop-optimizations">,
    Group<clang_ignored_gcc_optimization_f_Group>;
defm unswitch_loops : BooleanFFlag<"unswitch-loops">, Group<clang_ignored_gcc_optimization_f_Group>;
defm use_linker_plugin : BooleanFFlag<"use-linker-plugin">, Group<clang_ignored_gcc_optimization_f_Group>;
defm vect_cost_model : BooleanFFlag<"vect-cost-model">, Group<clang_ignored_gcc_optimization_f_Group>;
defm variable_expansion_in_unroller : BooleanFFlag<"variable-expansion-in-unroller">,
    Group<clang_ignored_gcc_optimization_f_Group>;
defm web : BooleanFFlag<"web">, Group<clang_ignored_gcc_optimization_f_Group>;
defm whole_program : BooleanFFlag<"whole-program">, Group<clang_ignored_gcc_optimization_f_Group>;
defm devirtualize : BooleanFFlag<"devirtualize">, Group<clang_ignored_gcc_optimization_f_Group>;
defm devirtualize_speculatively : BooleanFFlag<"devirtualize-speculatively">,
    Group<clang_ignored_gcc_optimization_f_Group>;

// Generic gfortran options.
def A_DASH : Joined<["-"], "A-">, Group<gfortran_Group>;
def static_libgfortran : Flag<["-"], "static-libgfortran">, Group<gfortran_Group>;

// "f" options with values for gfortran.
def fblas_matmul_limit_EQ : Joined<["-"], "fblas-matmul-limit=">, Group<gfortran_Group>;
def fcheck_EQ : Joined<["-"], "fcheck=">, Group<gfortran_Group>;
def fcoarray_EQ : Joined<["-"], "fcoarray=">, Group<gfortran_Group>;
def fconvert_EQ : Joined<["-"], "fconvert=">, Group<gfortran_Group>;
def ffpe_trap_EQ : Joined<["-"], "ffpe-trap=">, Group<gfortran_Group>;
def ffree_line_length_VALUE : Joined<["-"], "ffree-line-length-">, Group<gfortran_Group>;
def finit_character_EQ : Joined<["-"], "finit-character=">, Group<gfortran_Group>;
def finit_integer_EQ : Joined<["-"], "finit-integer=">, Group<gfortran_Group>;
def finit_logical_EQ : Joined<["-"], "finit-logical=">, Group<gfortran_Group>;
def finit_real_EQ : Joined<["-"], "finit-real=">, Group<gfortran_Group>;
def fmax_array_constructor_EQ : Joined<["-"], "fmax-array-constructor=">, Group<gfortran_Group>;
def fmax_errors_EQ : Joined<["-"], "fmax-errors=">, Group<gfortran_Group>;
def fmax_stack_var_size_EQ : Joined<["-"], "fmax-stack-var-size=">, Group<gfortran_Group>;
def fmax_subrecord_length_EQ : Joined<["-"], "fmax-subrecord-length=">, Group<gfortran_Group>;
def frecord_marker_EQ : Joined<["-"], "frecord-marker=">, Group<gfortran_Group>;

// "f" flags for gfortran.
defm aggressive_function_elimination : BooleanFFlag<"aggressive-function-elimination">, Group<gfortran_Group>;
defm align_commons : BooleanFFlag<"align-commons">, Group<gfortran_Group>;
defm all_intrinsics : BooleanFFlag<"all-intrinsics">, Group<gfortran_Group>;
def fautomatic : Flag<["-"], "fautomatic">; // -fno-automatic is significant
defm backtrace : BooleanFFlag<"backtrace">, Group<gfortran_Group>;
defm bounds_check : BooleanFFlag<"bounds-check">, Group<gfortran_Group>;
defm check_array_temporaries : BooleanFFlag<"check-array-temporaries">, Group<gfortran_Group>;
defm cray_pointer : BooleanFFlag<"cray-pointer">, Group<gfortran_Group>;
defm d_lines_as_code : BooleanFFlag<"d-lines-as-code">, Group<gfortran_Group>;
defm d_lines_as_comments : BooleanFFlag<"d-lines-as-comments">, Group<gfortran_Group>;
defm dollar_ok : BooleanFFlag<"dollar-ok">, Group<gfortran_Group>;
defm dump_fortran_optimized : BooleanFFlag<"dump-fortran-optimized">, Group<gfortran_Group>;
defm dump_fortran_original : BooleanFFlag<"dump-fortran-original">, Group<gfortran_Group>;
defm dump_parse_tree : BooleanFFlag<"dump-parse-tree">, Group<gfortran_Group>;
defm external_blas : BooleanFFlag<"external-blas">, Group<gfortran_Group>;
defm f2c : BooleanFFlag<"f2c">, Group<gfortran_Group>;
defm frontend_optimize : BooleanFFlag<"frontend-optimize">, Group<gfortran_Group>;
defm init_local_zero : BooleanFFlag<"init-local-zero">, Group<gfortran_Group>;
defm integer_4_integer_8 : BooleanFFlag<"integer-4-integer-8">, Group<gfortran_Group>;
defm max_identifier_length : BooleanFFlag<"max-identifier-length">, Group<gfortran_Group>;
defm module_private : BooleanFFlag<"module-private">, Group<gfortran_Group>;
defm pack_derived : BooleanFFlag<"pack-derived">, Group<gfortran_Group>;
//defm protect_parens : BooleanFFlag<"protect-parens">, Group<gfortran_Group>;
defm range_check : BooleanFFlag<"range-check">, Group<gfortran_Group>;
defm real_4_real_10 : BooleanFFlag<"real-4-real-10">, Group<gfortran_Group>;
defm real_4_real_16 : BooleanFFlag<"real-4-real-16">, Group<gfortran_Group>;
defm real_4_real_8 : BooleanFFlag<"real-4-real-8">, Group<gfortran_Group>;
defm real_8_real_10 : BooleanFFlag<"real-8-real-10">, Group<gfortran_Group>;
defm real_8_real_16 : BooleanFFlag<"real-8-real-16">, Group<gfortran_Group>;
defm real_8_real_4 : BooleanFFlag<"real-8-real-4">, Group<gfortran_Group>;
defm realloc_lhs : BooleanFFlag<"realloc-lhs">, Group<gfortran_Group>;
defm recursive : BooleanFFlag<"recursive">, Group<gfortran_Group>;
defm repack_arrays : BooleanFFlag<"repack-arrays">, Group<gfortran_Group>;
defm second_underscore : BooleanFFlag<"second-underscore">, Group<gfortran_Group>;
defm sign_zero : BooleanFFlag<"sign-zero">, Group<gfortran_Group>;
defm stack_arrays : BooleanFFlag<"stack-arrays">, Group<gfortran_Group>;
defm underscoring : BooleanFFlag<"underscoring">, Group<gfortran_Group>;
defm whole_file : BooleanFFlag<"whole-file">, Group<gfortran_Group>;

// C++ SYCL options
def fsycl : Flag<["-"], "fsycl">, Flags<[NoXarchOption, CoreOption]>,
  Group<sycl_Group>, HelpText<"Enables SYCL kernels compilation for device">;
def fno_sycl : Flag<["-"], "fno-sycl">, Flags<[NoXarchOption, CoreOption]>,
  Group<sycl_Group>, HelpText<"Disables SYCL kernels compilation for device">;

//===----------------------------------------------------------------------===//
// FLangOption + NoXarchOption
//===----------------------------------------------------------------------===//

def flang_experimental_exec : Flag<["-"], "flang-experimental-exec">,
  Flags<[FlangOption, FlangOnlyOption, NoXarchOption, HelpHidden]>,
  HelpText<"Enable support for generating executables (experimental)">;

//===----------------------------------------------------------------------===//
// FLangOption + CoreOption + NoXarchOption
//===----------------------------------------------------------------------===//

def Xflang : Separate<["-"], "Xflang">,
  HelpText<"Pass <arg> to the flang compiler">, MetaVarName<"<arg>">,
  Flags<[FlangOption, FlangOnlyOption, NoXarchOption, CoreOption]>,
  Group<CompileOnly_Group>;

//===----------------------------------------------------------------------===//
// FlangOption and FC1 Options
//===----------------------------------------------------------------------===//

let Flags = [FC1Option, FlangOption, FlangOnlyOption] in {

def cpp : Flag<["-"], "cpp">, Group<f_Group>,
  HelpText<"Enable predefined and command line preprocessor macros">;
def nocpp : Flag<["-"], "nocpp">, Group<f_Group>,
  HelpText<"Disable predefined and command line preprocessor macros">;
def module_dir : JoinedOrSeparate<["-"], "module-dir">, MetaVarName<"<dir>">,
  HelpText<"Put MODULE files in <dir>">,
  DocBrief<[{This option specifies where to put .mod files for compiled modules.
It is also added to the list of directories to be searched by an USE statement.
The default is the current directory.}]>;

def ffixed_form : Flag<["-"], "ffixed-form">, Group<f_Group>,
  HelpText<"Process source files in fixed form">;
def ffree_form : Flag<["-"], "ffree-form">, Group<f_Group>,
  HelpText<"Process source files in free form">;
def ffixed_line_length_EQ : Joined<["-"], "ffixed-line-length=">, Group<f_Group>,
  HelpText<"Use <value> as character line width in fixed mode">,
  DocBrief<[{Set column after which characters are ignored in typical fixed-form lines in the source
file}]>;
def ffixed_line_length_VALUE : Joined<["-"], "ffixed-line-length-">, Group<f_Group>, Alias<ffixed_line_length_EQ>;
def fopenacc : Flag<["-"], "fopenacc">, Group<f_Group>,
  HelpText<"Enable OpenACC">;
def fdefault_double_8 : Flag<["-"],"fdefault-double-8">, Group<f_Group>,
  HelpText<"Set the default double precision kind to an 8 byte wide type">;
def fdefault_integer_8 : Flag<["-"],"fdefault-integer-8">, Group<f_Group>,
  HelpText<"Set the default integer kind to an 8 byte wide type">;
def fdefault_real_8 : Flag<["-"],"fdefault-real-8">, Group<f_Group>,
  HelpText<"Set the default real kind to an 8 byte wide type">;
def flarge_sizes : Flag<["-"],"flarge-sizes">, Group<f_Group>,
  HelpText<"Use INTEGER(KIND=8) for the result type in size-related intrinsics">;

def falternative_parameter_statement : Flag<["-"], "falternative-parameter-statement">, Group<f_Group>,
  HelpText<"Enable the old style PARAMETER statement">;
def fintrinsic_modules_path : Separate<["-"], "fintrinsic-modules-path">,  Group<f_Group>, MetaVarName<"<dir>">,
  HelpText<"Specify where to find the compiled intrinsic modules">,
  DocBrief<[{This option specifies the location of pre-compiled intrinsic modules,
  if they are not in the default location expected by the compiler.}]>;

defm backslash : OptInFC1FFlag<"backslash", "Specify that backslash in string introduces an escape character">;
defm xor_operator : OptInFC1FFlag<"xor-operator", "Enable .XOR. as a synonym of .NEQV.">;
defm logical_abbreviations : OptInFC1FFlag<"logical-abbreviations", "Enable logical abbreviations">;
defm implicit_none : OptInFC1FFlag<"implicit-none", "No implicit typing allowed unless overridden by IMPLICIT statements">;

def fno_automatic : Flag<["-"], "fno-automatic">, Group<f_Group>,
  HelpText<"Implies the SAVE attribute for non-automatic local objects in subprograms unless RECURSIVE">;

} // let Flags = [FC1Option, FlangOption, FlangOnlyOption]

def J : JoinedOrSeparate<["-"], "J">,
  Flags<[RenderJoined, FlangOption, FC1Option, FlangOnlyOption]>,
  Group<gfortran_Group>,
  Alias<module_dir>;

//===----------------------------------------------------------------------===//
// FC1 Options
//===----------------------------------------------------------------------===//

let Flags = [FC1Option, FlangOnlyOption] in {

def fget_definition : MultiArg<["-"], "fget-definition", 3>,
  HelpText<"Get the symbol definition from <line> <start-column> <end-column>">,
  Group<Action_Group>;
def test_io : Flag<["-"], "test-io">, Group<Action_Group>,
  HelpText<"Run the InputOuputTest action. Use for development and testing only.">;
def fdebug_unparse_no_sema : Flag<["-"], "fdebug-unparse-no-sema">, Group<Action_Group>,
  HelpText<"Unparse and stop (skips the semantic checks)">,
  DocBrief<[{Only run the parser, then unparse the parse-tree and output the
generated Fortran source file. Semantic checks are disabled.}]>;
def fdebug_unparse : Flag<["-"], "fdebug-unparse">, Group<Action_Group>,
  HelpText<"Unparse and stop.">,
  DocBrief<[{Run the parser and the semantic checks. Then unparse the
parse-tree and output the generated Fortran source file.}]>;
def fdebug_unparse_with_symbols : Flag<["-"], "fdebug-unparse-with-symbols">, Group<Action_Group>,
  HelpText<"Unparse and stop.">;
def fdebug_dump_symbols : Flag<["-"], "fdebug-dump-symbols">, Group<Action_Group>,
  HelpText<"Dump symbols after the semantic analysis">;
def fdebug_dump_parse_tree : Flag<["-"], "fdebug-dump-parse-tree">, Group<Action_Group>,
  HelpText<"Dump the parse tree">,
  DocBrief<[{Run the Parser and the semantic checks, and then output the
parse tree.}]>;
def fdebug_dump_pft : Flag<["-"], "fdebug-dump-pft">, Group<Action_Group>,
  HelpText<"Dump the pre-fir parse tree">;
def fdebug_dump_parse_tree_no_sema : Flag<["-"], "fdebug-dump-parse-tree-no-sema">, Group<Action_Group>,
  HelpText<"Dump the parse tree (skips the semantic checks)">,
  DocBrief<[{Run the Parser and then output the parse tree. Semantic
checks are disabled.}]>;
def fdebug_dump_all : Flag<["-"], "fdebug-dump-all">, Group<Action_Group>,
  HelpText<"Dump symbols and the parse tree after the semantic checks">;
def fdebug_dump_provenance : Flag<["-"], "fdebug-dump-provenance">, Group<Action_Group>,
  HelpText<"Dump provenance">;
def fdebug_dump_parsing_log : Flag<["-"], "fdebug-dump-parsing-log">, Group<Action_Group>,
  HelpText<"Run instrumented parse and dump the parsing log">;
def fdebug_measure_parse_tree : Flag<["-"], "fdebug-measure-parse-tree">, Group<Action_Group>,
  HelpText<"Measure the parse tree">;
def fdebug_pre_fir_tree : Flag<["-"], "fdebug-pre-fir-tree">, Group<Action_Group>,
  HelpText<"Dump the pre-FIR tree">;
def fdebug_module_writer : Flag<["-"],"fdebug-module-writer">,
  HelpText<"Enable debug messages while writing module files">;
def fget_symbols_sources : Flag<["-"], "fget-symbols-sources">, Group<Action_Group>,
  HelpText<"Dump symbols and their source code locations">;

def module_suffix : Separate<["-"], "module-suffix">,  Group<f_Group>, MetaVarName<"<suffix>">,
  HelpText<"Use <suffix> as the suffix for module files (the default value is `.mod`)">;
def fno_reformat : Flag<["-"], "fno-reformat">, Group<Preprocessor_Group>,
  HelpText<"Dump the cooked character stream in -E mode">;
defm analyzed_objects_for_unparse : OptOutFC1FFlag<"analyzed-objects-for-unparse", "", "Do not use the analyzed objects when unparsing">;

def emit_mlir : Flag<["-"], "emit-mlir">, Group<Action_Group>,
  HelpText<"Build the parse tree, then lower it to MLIR">;
def emit_fir : Flag<["-"], "emit-fir">, Alias<emit_mlir>;

} // let Flags = [FC1Option, FlangOnlyOption]

//===----------------------------------------------------------------------===//
// Target Options (cc1 + cc1as)
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, CC1AsOption, NoDriverOption] in {

def target_cpu : Separate<["-"], "target-cpu">,
  HelpText<"Target a specific cpu type">,
  MarshallingInfoString<TargetOpts<"CPU">>;
def tune_cpu : Separate<["-"], "tune-cpu">,
  HelpText<"Tune for a specific cpu type">,
  MarshallingInfoString<TargetOpts<"TuneCPU">>;
def target_feature : Separate<["-"], "target-feature">,
  HelpText<"Target specific attributes">,
  MarshallingInfoStringVector<TargetOpts<"FeaturesAsWritten">>;
def target_abi : Separate<["-"], "target-abi">,
  HelpText<"Target a particular ABI type">,
  MarshallingInfoString<TargetOpts<"ABI">>;
def target_sdk_version_EQ : Joined<["-"], "target-sdk-version=">,
  HelpText<"The version of target SDK used for compilation">;
def darwin_target_variant_sdk_version_EQ : Joined<["-"],
  "darwin-target-variant-sdk-version=">,
  HelpText<"The version of darwin target variant SDK used for compilation">;

} // let Flags = [CC1Option, CC1AsOption, NoDriverOption]

let Flags = [CC1Option, CC1AsOption] in {

def darwin_target_variant_triple : Separate<["-"], "darwin-target-variant-triple">,
  HelpText<"Specify the darwin target variant triple">,
  MarshallingInfoString<TargetOpts<"DarwinTargetVariantTriple">>,
  Normalizer<"normalizeTriple">;

} // let Flags = [CC1Option, CC1AsOption]

//===----------------------------------------------------------------------===//
// Target Options (cc1 + cc1as + fc1)
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, CC1AsOption, FC1Option, NoDriverOption] in {

def triple : Separate<["-"], "triple">,
  HelpText<"Specify target triple (e.g. i686-apple-darwin9)">,
  MarshallingInfoString<TargetOpts<"Triple">, "llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple())">,
  AlwaysEmit, Normalizer<"normalizeTriple">;

} // let Flags = [CC1Option, CC1ASOption, FC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// Target Options (other)
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, NoDriverOption] in {

def target_linker_version : Separate<["-"], "target-linker-version">,
  HelpText<"Target linker version">,
  MarshallingInfoString<TargetOpts<"LinkerVersion">>;
def triple_EQ : Joined<["-"], "triple=">, Alias<triple>;
def mfpmath : Separate<["-"], "mfpmath">,
  HelpText<"Which unit to use for fp math">,
  MarshallingInfoString<TargetOpts<"FPMath">>;

defm padding_on_unsigned_fixed_point : BoolOption<"f", "padding-on-unsigned-fixed-point",
  LangOpts<"PaddingOnUnsignedFixedPoint">, DefaultFalse,
  PosFlag<SetTrue, [], "Force each unsigned fixed point type to have an extra bit of padding to align their scales with those of signed fixed point types">,
  NegFlag<SetFalse>>,
  ShouldParseIf<ffixed_point.KeyPath>;

} // let Flags = [CC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// Analyzer Options
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, NoDriverOption] in {

def analysis_UnoptimizedCFG : Flag<["-"], "unoptimized-cfg">,
  HelpText<"Generate unoptimized CFGs for all analyses">,
  MarshallingInfoFlag<AnalyzerOpts<"UnoptimizedCFG">>;
def analysis_CFGAddImplicitDtors : Flag<["-"], "cfg-add-implicit-dtors">,
  HelpText<"Add C++ implicit destructors to CFGs for all analyses">;

// We should remove this option in clang-16 release.
def analyzer_store : Separate<["-"], "analyzer-store">,
  HelpText<"Source Code Analysis - Abstract Memory Store Models [DEPRECATED, removing in clang-16]">;
def analyzer_store_EQ : Joined<["-"], "analyzer-store=">, Alias<analyzer_store>;

def analyzer_constraints : Separate<["-"], "analyzer-constraints">,
  HelpText<"Source Code Analysis - Symbolic Constraint Engines">;
def analyzer_constraints_EQ : Joined<["-"], "analyzer-constraints=">,
  Alias<analyzer_constraints>;

def analyzer_output : Separate<["-"], "analyzer-output">,
  HelpText<"Source Code Analysis - Output Options">;
def analyzer_output_EQ : Joined<["-"], "analyzer-output=">,
  Alias<analyzer_output>;

def analyzer_purge : Separate<["-"], "analyzer-purge">,
  HelpText<"Source Code Analysis - Dead Symbol Removal Frequency">;
def analyzer_purge_EQ : Joined<["-"], "analyzer-purge=">, Alias<analyzer_purge>;

def analyzer_opt_analyze_headers : Flag<["-"], "analyzer-opt-analyze-headers">,
  HelpText<"Force the static analyzer to analyze functions defined in header files">,
  MarshallingInfoFlag<AnalyzerOpts<"AnalyzeAll">>;
// We should remove this option in clang-16 release.
def analyzer_opt_analyze_nested_blocks : Flag<["-"], "analyzer-opt-analyze-nested-blocks">,
  HelpText<"Analyze the definitions of blocks in addition to functions [DEPRECATED, removing in clang-16]">;
def analyzer_display_progress : Flag<["-"], "analyzer-display-progress">,
  HelpText<"Emit verbose output about the analyzer's progress">,
  MarshallingInfoFlag<AnalyzerOpts<"AnalyzerDisplayProgress">>;
def analyze_function : Separate<["-"], "analyze-function">,
  HelpText<"Run analysis on specific function (for C++ include parameters in name)">,
  MarshallingInfoString<AnalyzerOpts<"AnalyzeSpecificFunction">>;
def analyze_function_EQ : Joined<["-"], "analyze-function=">, Alias<analyze_function>;
def trim_egraph : Flag<["-"], "trim-egraph">,
  HelpText<"Only show error-related paths in the analysis graph">,
  MarshallingInfoFlag<AnalyzerOpts<"TrimGraph">>;
def analyzer_viz_egraph_graphviz : Flag<["-"], "analyzer-viz-egraph-graphviz">,
  HelpText<"Display exploded graph using GraphViz">,
  MarshallingInfoFlag<AnalyzerOpts<"visualizeExplodedGraphWithGraphViz">>;
def analyzer_dump_egraph : Separate<["-"], "analyzer-dump-egraph">,
  HelpText<"Dump exploded graph to the specified file">,
  MarshallingInfoString<AnalyzerOpts<"DumpExplodedGraphTo">>;
def analyzer_dump_egraph_EQ : Joined<["-"], "analyzer-dump-egraph=">, Alias<analyzer_dump_egraph>;

def analyzer_inline_max_stack_depth : Separate<["-"], "analyzer-inline-max-stack-depth">,
  HelpText<"Bound on stack depth while inlining (4 by default)">,
  // Cap the stack depth at 4 calls (5 stack frames, base + 4 calls).
  MarshallingInfoInt<AnalyzerOpts<"InlineMaxStackDepth">, "5">;
def analyzer_inline_max_stack_depth_EQ : Joined<["-"], "analyzer-inline-max-stack-depth=">,
  Alias<analyzer_inline_max_stack_depth>;

def analyzer_inlining_mode : Separate<["-"], "analyzer-inlining-mode">,
  HelpText<"Specify the function selection heuristic used during inlining">;
def analyzer_inlining_mode_EQ : Joined<["-"], "analyzer-inlining-mode=">, Alias<analyzer_inlining_mode>;

def analyzer_disable_retry_exhausted : Flag<["-"], "analyzer-disable-retry-exhausted">,
  HelpText<"Do not re-analyze paths leading to exhausted nodes with a different strategy (may decrease code coverage)">,
  MarshallingInfoFlag<AnalyzerOpts<"NoRetryExhausted">>;

def analyzer_max_loop : Separate<["-"], "analyzer-max-loop">,
  HelpText<"The maximum number of times the analyzer will go through a loop">,
  MarshallingInfoInt<AnalyzerOpts<"maxBlockVisitOnPath">, "4">;
def analyzer_stats : Flag<["-"], "analyzer-stats">,
  HelpText<"Print internal analyzer statistics.">,
  MarshallingInfoFlag<AnalyzerOpts<"PrintStats">>;

def analyzer_checker : Separate<["-"], "analyzer-checker">,
  HelpText<"Choose analyzer checkers to enable">,
  ValuesCode<[{
    const char *Values =
    #define GET_CHECKERS
    #define CHECKER(FULLNAME, CLASS, HT, DOC_URI, IS_HIDDEN)  FULLNAME ","
    #include "clang/StaticAnalyzer/Checkers/Checkers.inc"
    #undef GET_CHECKERS
    #define GET_PACKAGES
    #define PACKAGE(FULLNAME)  FULLNAME ","
    #include "clang/StaticAnalyzer/Checkers/Checkers.inc"
    #undef GET_PACKAGES
    ;
  }]>;
def analyzer_checker_EQ : Joined<["-"], "analyzer-checker=">,
  Alias<analyzer_checker>;

def analyzer_disable_checker : Separate<["-"], "analyzer-disable-checker">,
  HelpText<"Choose analyzer checkers to disable">;
def analyzer_disable_checker_EQ : Joined<["-"], "analyzer-disable-checker=">,
  Alias<analyzer_disable_checker>;

def analyzer_disable_all_checks : Flag<["-"], "analyzer-disable-all-checks">,
  HelpText<"Disable all static analyzer checks">,
  MarshallingInfoFlag<AnalyzerOpts<"DisableAllCheckers">>;

def analyzer_checker_help : Flag<["-"], "analyzer-checker-help">,
  HelpText<"Display the list of analyzer checkers that are available">,
  MarshallingInfoFlag<AnalyzerOpts<"ShowCheckerHelp">>;

def analyzer_checker_help_alpha : Flag<["-"], "analyzer-checker-help-alpha">,
  HelpText<"Display the list of in development analyzer checkers. These "
           "are NOT considered safe, they are unstable and will emit incorrect "
           "reports. Enable ONLY FOR DEVELOPMENT purposes">,
  MarshallingInfoFlag<AnalyzerOpts<"ShowCheckerHelpAlpha">>;

def analyzer_checker_help_developer : Flag<["-"], "analyzer-checker-help-developer">,
  HelpText<"Display the list of developer-only checkers such as modeling "
           "and debug checkers">,
  MarshallingInfoFlag<AnalyzerOpts<"ShowCheckerHelpDeveloper">>;

def analyzer_config_help : Flag<["-"], "analyzer-config-help">,
  HelpText<"Display the list of -analyzer-config options. These are meant for "
           "development purposes only!">,
  MarshallingInfoFlag<AnalyzerOpts<"ShowConfigOptionsList">>;

def analyzer_list_enabled_checkers : Flag<["-"], "analyzer-list-enabled-checkers">,
  HelpText<"Display the list of enabled analyzer checkers">,
  MarshallingInfoFlag<AnalyzerOpts<"ShowEnabledCheckerList">>;

def analyzer_config : Separate<["-"], "analyzer-config">,
  HelpText<"Choose analyzer options to enable">;

def analyzer_checker_option_help : Flag<["-"], "analyzer-checker-option-help">,
  HelpText<"Display the list of checker and package options">,
  MarshallingInfoFlag<AnalyzerOpts<"ShowCheckerOptionList">>;

def analyzer_checker_option_help_alpha : Flag<["-"], "analyzer-checker-option-help-alpha">,
  HelpText<"Display the list of in development checker and package options. "
           "These are NOT considered safe, they are unstable and will emit "
           "incorrect reports. Enable ONLY FOR DEVELOPMENT purposes">,
  MarshallingInfoFlag<AnalyzerOpts<"ShowCheckerOptionAlphaList">>;

def analyzer_checker_option_help_developer : Flag<["-"], "analyzer-checker-option-help-developer">,
  HelpText<"Display the list of checker and package options meant for "
           "development purposes only">,
  MarshallingInfoFlag<AnalyzerOpts<"ShowCheckerOptionDeveloperList">>;

def analyzer_config_compatibility_mode : Separate<["-"], "analyzer-config-compatibility-mode">,
  HelpText<"Don't emit errors on invalid analyzer-config inputs">,
  Values<"true,false">, NormalizedValues<[[{false}], [{true}]]>,
  MarshallingInfoEnum<AnalyzerOpts<"ShouldEmitErrorsOnInvalidConfigValue">, [{true}]>;

def analyzer_config_compatibility_mode_EQ : Joined<["-"], "analyzer-config-compatibility-mode=">,
  Alias<analyzer_config_compatibility_mode>;

def analyzer_werror : Flag<["-"], "analyzer-werror">,
  HelpText<"Emit analyzer results as errors rather than warnings">,
  MarshallingInfoFlag<AnalyzerOpts<"AnalyzerWerror">>;

} // let Flags = [CC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// Migrator Options
//===----------------------------------------------------------------------===//

def migrator_no_nsalloc_error : Flag<["-"], "no-ns-alloc-error">,
  HelpText<"Do not error on use of NSAllocateCollectable/NSReallocateCollectable">,
  Flags<[CC1Option, NoDriverOption]>,
  MarshallingInfoFlag<MigratorOpts<"NoNSAllocReallocError">>;

def migrator_no_finalize_removal : Flag<["-"], "no-finalize-removal">,
  HelpText<"Do not remove finalize method in gc mode">,
  Flags<[CC1Option, NoDriverOption]>,
  MarshallingInfoFlag<MigratorOpts<"NoFinalizeRemoval">>;

//===----------------------------------------------------------------------===//
// CodeGen Options
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, CC1AsOption, NoDriverOption] in {

def debug_info_kind_EQ : Joined<["-"], "debug-info-kind=">;
def debug_info_macro : Flag<["-"], "debug-info-macro">,
  HelpText<"Emit macro debug information">,
  MarshallingInfoFlag<CodeGenOpts<"MacroDebugInfo">>;
def default_function_attr : Separate<["-"], "default-function-attr">,
  HelpText<"Apply given attribute to all functions">,
  MarshallingInfoStringVector<CodeGenOpts<"DefaultFunctionAttrs">>;
def dwarf_version_EQ : Joined<["-"], "dwarf-version=">,
  MarshallingInfoInt<CodeGenOpts<"DwarfVersion">>;
def debugger_tuning_EQ : Joined<["-"], "debugger-tuning=">,
  Values<"gdb,lldb,sce,dbx">,
  NormalizedValuesScope<"llvm::DebuggerKind">, NormalizedValues<["GDB", "LLDB", "SCE", "DBX"]>,
  MarshallingInfoEnum<CodeGenOpts<"DebuggerTuning">, "Default">;
def dwarf_debug_flags : Separate<["-"], "dwarf-debug-flags">,
  HelpText<"The string to embed in the Dwarf debug flags record.">,
  MarshallingInfoString<CodeGenOpts<"DwarfDebugFlags">>;
def record_command_line : Separate<["-"], "record-command-line">,
  HelpText<"The string to embed in the .LLVM.command.line section.">,
  MarshallingInfoString<CodeGenOpts<"RecordCommandLine">>;
def compress_debug_sections_EQ : Joined<["-", "--"], "compress-debug-sections=">,
    HelpText<"DWARF debug sections compression type">, Values<"none,zlib">,
    NormalizedValuesScope<"llvm::DebugCompressionType">, NormalizedValues<["None", "Z"]>,
    MarshallingInfoEnum<CodeGenOpts<"CompressDebugSections">, "None">;
def compress_debug_sections : Flag<["-", "--"], "compress-debug-sections">,
  Alias<compress_debug_sections_EQ>, AliasArgs<["zlib"]>;
def mno_exec_stack : Flag<["-"], "mnoexecstack">,
  HelpText<"Mark the file as not needing an executable stack">,
  MarshallingInfoFlag<CodeGenOpts<"NoExecStack">>;
def massembler_no_warn : Flag<["-"], "massembler-no-warn">,
  HelpText<"Make assembler not emit warnings">,
  MarshallingInfoFlag<CodeGenOpts<"NoWarn">>;
def massembler_fatal_warnings : Flag<["-"], "massembler-fatal-warnings">,
  HelpText<"Make assembler warnings fatal">,
  MarshallingInfoFlag<CodeGenOpts<"FatalWarnings">>;
def mrelax_relocations : Flag<["--"], "mrelax-relocations">,
    HelpText<"Use relaxable elf relocations">,
    MarshallingInfoFlag<CodeGenOpts<"RelaxELFRelocations">>;
def msave_temp_labels : Flag<["-"], "msave-temp-labels">,
  HelpText<"Save temporary labels in the symbol table. "
           "Note this may change .s semantics and shouldn't generally be used "
           "on compiler-generated code.">,
  MarshallingInfoFlag<CodeGenOpts<"SaveTempLabels">>;
def mrelocation_model : Separate<["-"], "mrelocation-model">,
  HelpText<"The relocation model to use">, Values<"static,pic,ropi,rwpi,ropi-rwpi,dynamic-no-pic">,
  NormalizedValuesScope<"llvm::Reloc">,
  NormalizedValues<["Static", "PIC_", "ROPI", "RWPI", "ROPI_RWPI", "DynamicNoPIC"]>,
  MarshallingInfoEnum<CodeGenOpts<"RelocationModel">, "PIC_">;
def fno_math_builtin : Flag<["-"], "fno-math-builtin">,
  HelpText<"Disable implicit builtin knowledge of math functions">,
  MarshallingInfoFlag<LangOpts<"NoMathBuiltin">>;
def fno_use_ctor_homing: Flag<["-"], "fno-use-ctor-homing">,
    HelpText<"Don't use constructor homing for debug info">;
def fuse_ctor_homing: Flag<["-"], "fuse-ctor-homing">,
    HelpText<"Use constructor homing if we are using limited debug info already">;

} // let Flags = [CC1Option, CC1AsOption, NoDriverOption]

let Flags = [CC1Option, NoDriverOption] in {

def disable_llvm_verifier : Flag<["-"], "disable-llvm-verifier">,
  HelpText<"Don't run the LLVM IR verifier pass">,
  MarshallingInfoNegativeFlag<CodeGenOpts<"VerifyModule">>;
def disable_llvm_passes : Flag<["-"], "disable-llvm-passes">,
  HelpText<"Use together with -emit-llvm to get pristine LLVM IR from the "
           "frontend by not running any LLVM passes at all">,
  MarshallingInfoFlag<CodeGenOpts<"DisableLLVMPasses">>;
def disable_llvm_optzns : Flag<["-"], "disable-llvm-optzns">,
  Alias<disable_llvm_passes>;
def disable_lifetimemarkers : Flag<["-"], "disable-lifetime-markers">,
  HelpText<"Disable lifetime-markers emission even when optimizations are "
           "enabled">,
  MarshallingInfoFlag<CodeGenOpts<"DisableLifetimeMarkers">>;
def disable_O0_optnone : Flag<["-"], "disable-O0-optnone">,
  HelpText<"Disable adding the optnone attribute to functions at O0">,
  MarshallingInfoFlag<CodeGenOpts<"DisableO0ImplyOptNone">>;
def disable_red_zone : Flag<["-"], "disable-red-zone">,
  HelpText<"Do not emit code that uses the red zone.">,
  MarshallingInfoFlag<CodeGenOpts<"DisableRedZone">>;
def dwarf_ext_refs : Flag<["-"], "dwarf-ext-refs">,
  HelpText<"Generate debug info with external references to clang modules"
           " or precompiled headers">,
  MarshallingInfoFlag<CodeGenOpts<"DebugTypeExtRefs">>;
def dwarf_explicit_import : Flag<["-"], "dwarf-explicit-import">,
  HelpText<"Generate explicit import from anonymous namespace to containing"
           " scope">,
  MarshallingInfoFlag<CodeGenOpts<"DebugExplicitImport">>;
def debug_forward_template_params : Flag<["-"], "debug-forward-template-params">,
  HelpText<"Emit complete descriptions of template parameters in forward"
           " declarations">,
  MarshallingInfoFlag<CodeGenOpts<"DebugFwdTemplateParams">>;
def fforbid_guard_variables : Flag<["-"], "fforbid-guard-variables">,
  HelpText<"Emit an error if a C++ static local initializer would need a guard variable">,
  MarshallingInfoFlag<CodeGenOpts<"ForbidGuardVariables">>;
def no_implicit_float : Flag<["-"], "no-implicit-float">,
  HelpText<"Don't generate implicit floating point instructions">,
  MarshallingInfoFlag<CodeGenOpts<"NoImplicitFloat">>;
def fdump_vtable_layouts : Flag<["-"], "fdump-vtable-layouts">,
  HelpText<"Dump the layouts of all vtables that will be emitted in a translation unit">,
  MarshallingInfoFlag<LangOpts<"DumpVTableLayouts">>;
def fmerge_functions : Flag<["-"], "fmerge-functions">,
  HelpText<"Permit merging of identical functions when optimizing.">,
  MarshallingInfoFlag<CodeGenOpts<"MergeFunctions">>;
def coverage_data_file : Separate<["-"], "coverage-data-file">,
  HelpText<"Emit coverage data to this filename.">,
  MarshallingInfoString<CodeGenOpts<"CoverageDataFile">>,
  ShouldParseIf<!strconcat(fprofile_arcs.KeyPath, "||", ftest_coverage.KeyPath)>;
def coverage_data_file_EQ : Joined<["-"], "coverage-data-file=">,
  Alias<coverage_data_file>;
def coverage_notes_file : Separate<["-"], "coverage-notes-file">,
  HelpText<"Emit coverage notes to this filename.">,
  MarshallingInfoString<CodeGenOpts<"CoverageNotesFile">>,
  ShouldParseIf<!strconcat(fprofile_arcs.KeyPath, "||", ftest_coverage.KeyPath)>;
def coverage_notes_file_EQ : Joined<["-"], "coverage-notes-file=">,
  Alias<coverage_notes_file>;
def coverage_version_EQ : Joined<["-"], "coverage-version=">,
  HelpText<"Four-byte version string for gcov files.">;
def dump_coverage_mapping : Flag<["-"], "dump-coverage-mapping">,
  HelpText<"Dump the coverage mapping records, for testing">,
  MarshallingInfoFlag<CodeGenOpts<"DumpCoverageMapping">>;
def fuse_register_sized_bitfield_access: Flag<["-"], "fuse-register-sized-bitfield-access">,
  HelpText<"Use register sized accesses to bit-fields, when possible.">,
  MarshallingInfoFlag<CodeGenOpts<"UseRegisterSizedBitfieldAccess">>;
def relaxed_aliasing : Flag<["-"], "relaxed-aliasing">,
  HelpText<"Turn off Type Based Alias Analysis">,
  MarshallingInfoFlag<CodeGenOpts<"RelaxedAliasing">>;
def no_struct_path_tbaa : Flag<["-"], "no-struct-path-tbaa">,
  HelpText<"Turn off struct-path aware Type Based Alias Analysis">,
  MarshallingInfoNegativeFlag<CodeGenOpts<"StructPathTBAA">>;
def new_struct_path_tbaa : Flag<["-"], "new-struct-path-tbaa">,
  HelpText<"Enable enhanced struct-path aware Type Based Alias Analysis">;
def mdebug_pass : Separate<["-"], "mdebug-pass">,
  HelpText<"Enable additional debug output">,
  MarshallingInfoString<CodeGenOpts<"DebugPass">>;
def mframe_pointer_EQ : Joined<["-"], "mframe-pointer=">,
  HelpText<"Specify which frame pointers to retain.">, Values<"all,non-leaf,none">,
  NormalizedValuesScope<"CodeGenOptions::FramePointerKind">, NormalizedValues<["All", "NonLeaf", "None"]>,
  MarshallingInfoEnum<CodeGenOpts<"FramePointer">, "None">;
def menable_no_infinities : Flag<["-"], "menable-no-infs">,
  HelpText<"Allow optimization to assume there are no infinities.">,
  MarshallingInfoFlag<LangOpts<"NoHonorInfs">>, ImpliedByAnyOf<[ffinite_math_only.KeyPath]>;
def menable_no_nans : Flag<["-"], "menable-no-nans">,
  HelpText<"Allow optimization to assume there are no NaNs.">,
  MarshallingInfoFlag<LangOpts<"NoHonorNaNs">>, ImpliedByAnyOf<[ffinite_math_only.KeyPath]>;
def mreassociate : Flag<["-"], "mreassociate">,
  HelpText<"Allow reassociation transformations for floating-point instructions">,
  MarshallingInfoFlag<LangOpts<"AllowFPReassoc">>, ImpliedByAnyOf<[menable_unsafe_fp_math.KeyPath]>;
def mabi_EQ_ieeelongdouble : Flag<["-"], "mabi=ieeelongdouble">,
  HelpText<"Use IEEE 754 quadruple-precision for long double">,
  MarshallingInfoFlag<LangOpts<"PPCIEEELongDouble">>;
def mfloat_abi : Separate<["-"], "mfloat-abi">,
  HelpText<"The float ABI to use">,
  MarshallingInfoString<CodeGenOpts<"FloatABI">>;
def mtp : Separate<["-"], "mtp">,
  HelpText<"Mode for reading thread pointer">;
def mlimit_float_precision : Separate<["-"], "mlimit-float-precision">,
  HelpText<"Limit float precision to the given value">,
  MarshallingInfoString<CodeGenOpts<"LimitFloatPrecision">>;
def mregparm : Separate<["-"], "mregparm">,
  HelpText<"Limit the number of registers available for integer arguments">,
  MarshallingInfoInt<CodeGenOpts<"NumRegisterParameters">>;
def msmall_data_limit : Separate<["-"], "msmall-data-limit">,
  HelpText<"Put global and static data smaller than the limit into a special section">,
  MarshallingInfoInt<CodeGenOpts<"SmallDataLimit">>;
def funwind_tables_EQ : Joined<["-"], "funwind-tables=">,
  HelpText<"Generate unwinding tables for all functions">,
  MarshallingInfoInt<CodeGenOpts<"UnwindTables">>;
defm constructor_aliases : BoolOption<"m", "constructor-aliases",
  CodeGenOpts<"CXXCtorDtorAliases">, DefaultFalse,
  PosFlag<SetTrue, [], "Enable">, NegFlag<SetFalse, [], "Disable">,
  BothFlags<[CC1Option], " emitting complete constructors and destructors as aliases when possible">>;
def mlink_bitcode_file : Separate<["-"], "mlink-bitcode-file">,
  HelpText<"Link the given bitcode file before performing optimizations.">;
def mlink_builtin_bitcode : Separate<["-"], "mlink-builtin-bitcode">,
  HelpText<"Link and internalize needed symbols from the given bitcode file "
           "before performing optimizations.">;
def mlink_cuda_bitcode : Separate<["-"], "mlink-cuda-bitcode">,
  Alias<mlink_builtin_bitcode>;
def vectorize_loops : Flag<["-"], "vectorize-loops">,
  HelpText<"Run the Loop vectorization passes">,
  MarshallingInfoFlag<CodeGenOpts<"VectorizeLoop">>;
def vectorize_slp : Flag<["-"], "vectorize-slp">,
  HelpText<"Run the SLP vectorization passes">,
  MarshallingInfoFlag<CodeGenOpts<"VectorizeSLP">>;
def dependent_lib : Joined<["--"], "dependent-lib=">,
  HelpText<"Add dependent library">,
  MarshallingInfoStringVector<CodeGenOpts<"DependentLibraries">>;
def linker_option : Joined<["--"], "linker-option=">,
  HelpText<"Add linker option">,
  MarshallingInfoStringVector<CodeGenOpts<"LinkerOptions">>;
def fsanitize_coverage_type : Joined<["-"], "fsanitize-coverage-type=">,
                              HelpText<"Sanitizer coverage type">,
                              MarshallingInfoInt<CodeGenOpts<"SanitizeCoverageType">>;
def fsanitize_coverage_indirect_calls
    : Flag<["-"], "fsanitize-coverage-indirect-calls">,
      HelpText<"Enable sanitizer coverage for indirect calls">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageIndirectCalls">>;
def fsanitize_coverage_trace_bb
    : Flag<["-"], "fsanitize-coverage-trace-bb">,
      HelpText<"Enable basic block tracing in sanitizer coverage">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageTraceBB">>;
def fsanitize_coverage_trace_cmp
    : Flag<["-"], "fsanitize-coverage-trace-cmp">,
      HelpText<"Enable cmp instruction tracing in sanitizer coverage">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageTraceCmp">>;
def fsanitize_coverage_trace_div
    : Flag<["-"], "fsanitize-coverage-trace-div">,
      HelpText<"Enable div instruction tracing in sanitizer coverage">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageTraceDiv">>;
def fsanitize_coverage_trace_gep
    : Flag<["-"], "fsanitize-coverage-trace-gep">,
      HelpText<"Enable gep instruction tracing in sanitizer coverage">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageTraceGep">>;
def fsanitize_coverage_8bit_counters
    : Flag<["-"], "fsanitize-coverage-8bit-counters">,
      HelpText<"Enable frequency counters in sanitizer coverage">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverage8bitCounters">>;
def fsanitize_coverage_inline_8bit_counters
    : Flag<["-"], "fsanitize-coverage-inline-8bit-counters">,
      HelpText<"Enable inline 8-bit counters in sanitizer coverage">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageInline8bitCounters">>;
def fsanitize_coverage_inline_bool_flag
    : Flag<["-"], "fsanitize-coverage-inline-bool-flag">,
      HelpText<"Enable inline bool flag in sanitizer coverage">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageInlineBoolFlag">>;
def fsanitize_coverage_pc_table
    : Flag<["-"], "fsanitize-coverage-pc-table">,
      HelpText<"Create a table of coverage-instrumented PCs">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoveragePCTable">>;
def fsanitize_coverage_trace_pc
    : Flag<["-"], "fsanitize-coverage-trace-pc">,
      HelpText<"Enable PC tracing in sanitizer coverage">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageTracePC">>;
def fsanitize_coverage_trace_pc_guard
    : Flag<["-"], "fsanitize-coverage-trace-pc-guard">,
      HelpText<"Enable PC tracing with guard in sanitizer coverage">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageTracePCGuard">>;
def fsanitize_coverage_no_prune
    : Flag<["-"], "fsanitize-coverage-no-prune">,
      HelpText<"Disable coverage pruning (i.e. instrument all blocks/edges)">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageNoPrune">>;
def fsanitize_coverage_stack_depth
    : Flag<["-"], "fsanitize-coverage-stack-depth">,
      HelpText<"Enable max stack depth tracing">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageStackDepth">>;
def fsanitize_coverage_trace_loads
    : Flag<["-"], "fsanitize-coverage-trace-loads">,
      HelpText<"Enable tracing of loads">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageTraceLoads">>;
def fsanitize_coverage_trace_stores
    : Flag<["-"], "fsanitize-coverage-trace-stores">,
      HelpText<"Enable tracing of stores">,
      MarshallingInfoFlag<CodeGenOpts<"SanitizeCoverageTraceStores">>;
def fpatchable_function_entry_offset_EQ
    : Joined<["-"], "fpatchable-function-entry-offset=">, MetaVarName<"<M>">,
      HelpText<"Generate M NOPs before function entry">,
      MarshallingInfoInt<CodeGenOpts<"PatchableFunctionEntryOffset">>;
def fprofile_instrument_EQ : Joined<["-"], "fprofile-instrument=">,
    HelpText<"Enable PGO instrumentation">, Values<"none,clang,llvm,csllvm">,
    NormalizedValuesScope<"CodeGenOptions">,
    NormalizedValues<["ProfileNone", "ProfileClangInstr", "ProfileIRInstr", "ProfileCSIRInstr"]>,
    MarshallingInfoEnum<CodeGenOpts<"ProfileInstr">, "ProfileNone">;
def fprofile_instrument_path_EQ : Joined<["-"], "fprofile-instrument-path=">,
    HelpText<"Generate instrumented code to collect execution counts into "
             "<file> (overridden by LLVM_PROFILE_FILE env var)">,
    MarshallingInfoString<CodeGenOpts<"InstrProfileOutput">>;
def fprofile_instrument_use_path_EQ :
    Joined<["-"], "fprofile-instrument-use-path=">,
    HelpText<"Specify the profile path in PGO use compilation">,
    MarshallingInfoString<CodeGenOpts<"ProfileInstrumentUsePath">>;
def flto_visibility_public_std:
    Flag<["-"], "flto-visibility-public-std">,
    HelpText<"Use public LTO visibility for classes in std and stdext namespaces">,
    MarshallingInfoFlag<CodeGenOpts<"LTOVisibilityPublicStd">>;
defm lto_unit : BoolOption<"f", "lto-unit",
  CodeGenOpts<"LTOUnit">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Emit IR to support LTO unit features (CFI, whole program vtable opt)">,
  NegFlag<SetFalse>>;
def fverify_debuginfo_preserve
    : Flag<["-"], "fverify-debuginfo-preserve">,
      HelpText<"Enable Debug Info Metadata preservation testing in "
               "optimizations.">,
      MarshallingInfoFlag<CodeGenOpts<"EnableDIPreservationVerify">>;
def fverify_debuginfo_preserve_export
    : Joined<["-"], "fverify-debuginfo-preserve-export=">,
      MetaVarName<"<file>">,
      HelpText<"Export debug info (by testing original Debug Info) failures "
               "into specified (JSON) file (should be abs path as we use "
               "append mode to insert new JSON objects).">,
      MarshallingInfoString<CodeGenOpts<"DIBugsReportFilePath">>;
def fwarn_stack_size_EQ
    : Joined<["-"], "fwarn-stack-size=">,
      MarshallingInfoInt<CodeGenOpts<"WarnStackSize">, "UINT_MAX">;
// The driver option takes the key as a parameter to the -msign-return-address=
// and -mbranch-protection= options, but CC1 has a separate option so we
// don't have to parse the parameter twice.
def msign_return_address_key_EQ : Joined<["-"], "msign-return-address-key=">,
    Values<"a_key,b_key">;
def mbranch_target_enforce : Flag<["-"], "mbranch-target-enforce">,
  MarshallingInfoFlag<LangOpts<"BranchTargetEnforcement">>;
def fno_dllexport_inlines : Flag<["-"], "fno-dllexport-inlines">,
  MarshallingInfoNegativeFlag<LangOpts<"DllExportInlines">>;
def cfguard_no_checks : Flag<["-"], "cfguard-no-checks">,
    HelpText<"Emit Windows Control Flow Guard tables only (no checks)">,
    MarshallingInfoFlag<CodeGenOpts<"ControlFlowGuardNoChecks">>;
def cfguard : Flag<["-"], "cfguard">,
    HelpText<"Emit Windows Control Flow Guard tables and checks">,
    MarshallingInfoFlag<CodeGenOpts<"ControlFlowGuard">>;
def ehcontguard : Flag<["-"], "ehcontguard">,
    HelpText<"Emit Windows EH Continuation Guard tables">,
    MarshallingInfoFlag<CodeGenOpts<"EHContGuard">>;

def fdenormal_fp_math_f32_EQ : Joined<["-"], "fdenormal-fp-math-f32=">,
   Group<f_Group>;

} // let Flags = [CC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// Dependency Output Options
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, NoDriverOption] in {

def sys_header_deps : Flag<["-"], "sys-header-deps">,
  HelpText<"Include system headers in dependency output">,
  MarshallingInfoFlag<DependencyOutputOpts<"IncludeSystemHeaders">>;
def module_file_deps : Flag<["-"], "module-file-deps">,
  HelpText<"Include module files in dependency output">,
  MarshallingInfoFlag<DependencyOutputOpts<"IncludeModuleFiles">>;
def header_include_file : Separate<["-"], "header-include-file">,
  HelpText<"Filename (or -) to write header include output to">,
  MarshallingInfoString<DependencyOutputOpts<"HeaderIncludeOutputFile">>;
def show_includes : Flag<["--"], "show-includes">,
  HelpText<"Print cl.exe style /showIncludes to stdout">;

} // let Flags = [CC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, NoDriverOption] in {

def diagnostic_log_file : Separate<["-"], "diagnostic-log-file">,
  HelpText<"Filename (or -) to log diagnostics to">,
  MarshallingInfoString<DiagnosticOpts<"DiagnosticLogFile">>;
def diagnostic_serialized_file : Separate<["-"], "serialize-diagnostic-file">,
  MetaVarName<"<filename>">,
  HelpText<"File for serializing diagnostics in a binary format">;

def fdiagnostics_format : Separate<["-"], "fdiagnostics-format">,
  HelpText<"Change diagnostic formatting to match IDE and command line tools">,
  Values<"clang,msvc,vi">,
  NormalizedValuesScope<"DiagnosticOptions">, NormalizedValues<["Clang", "MSVC", "Vi"]>,
  MarshallingInfoEnum<DiagnosticOpts<"Format">, "Clang">;
def fdiagnostics_show_category : Separate<["-"], "fdiagnostics-show-category">,
  HelpText<"Print diagnostic category">,
  Values<"none,id,name">,
  NormalizedValues<["0", "1", "2"]>,
  MarshallingInfoEnum<DiagnosticOpts<"ShowCategories">, "0">;
def fno_diagnostics_use_presumed_location : Flag<["-"], "fno-diagnostics-use-presumed-location">,
  HelpText<"Ignore #line directives when displaying diagnostic locations">,
  MarshallingInfoNegativeFlag<DiagnosticOpts<"ShowPresumedLoc">>;
def ftabstop : Separate<["-"], "ftabstop">, MetaVarName<"<N>">,
  HelpText<"Set the tab stop distance.">,
  MarshallingInfoInt<DiagnosticOpts<"TabStop">, "DiagnosticOptions::DefaultTabStop">;
def ferror_limit : Separate<["-"], "ferror-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of errors to emit before stopping (0 = no limit).">,
  MarshallingInfoInt<DiagnosticOpts<"ErrorLimit">>;
def fmacro_backtrace_limit : Separate<["-"], "fmacro-backtrace-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of entries to print in a macro expansion backtrace (0 = no limit).">,
  MarshallingInfoInt<DiagnosticOpts<"MacroBacktraceLimit">, "DiagnosticOptions::DefaultMacroBacktraceLimit">;
def ftemplate_backtrace_limit : Separate<["-"], "ftemplate-backtrace-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of entries to print in a template instantiation backtrace (0 = no limit).">,
  MarshallingInfoInt<DiagnosticOpts<"TemplateBacktraceLimit">, "DiagnosticOptions::DefaultTemplateBacktraceLimit">;
def fconstexpr_backtrace_limit : Separate<["-"], "fconstexpr-backtrace-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of entries to print in a constexpr evaluation backtrace (0 = no limit).">,
  MarshallingInfoInt<DiagnosticOpts<"ConstexprBacktraceLimit">, "DiagnosticOptions::DefaultConstexprBacktraceLimit">;
def fspell_checking_limit : Separate<["-"], "fspell-checking-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of times to perform spell checking on unrecognized identifiers (0 = no limit).">,
  MarshallingInfoInt<DiagnosticOpts<"SpellCheckingLimit">, "DiagnosticOptions::DefaultSpellCheckingLimit">;
def fcaret_diagnostics_max_lines :
  Separate<["-"], "fcaret-diagnostics-max-lines">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of source lines to show in a caret diagnostic">,
  MarshallingInfoInt<DiagnosticOpts<"SnippetLineLimit">, "DiagnosticOptions::DefaultSnippetLineLimit">;
def verify_EQ : CommaJoined<["-"], "verify=">,
  MetaVarName<"<prefixes>">,
  HelpText<"Verify diagnostic output using comment directives that start with"
           " prefixes in the comma-separated sequence <prefixes>">;
def verify : Flag<["-"], "verify">,
  HelpText<"Equivalent to -verify=expected">;
def verify_ignore_unexpected : Flag<["-"], "verify-ignore-unexpected">,
  HelpText<"Ignore unexpected diagnostic messages">;
def verify_ignore_unexpected_EQ : CommaJoined<["-"], "verify-ignore-unexpected=">,
  HelpText<"Ignore unexpected diagnostic messages">;
def Wno_rewrite_macros : Flag<["-"], "Wno-rewrite-macros">,
  HelpText<"Silence ObjC rewriting warnings">,
  MarshallingInfoFlag<DiagnosticOpts<"NoRewriteMacros">>;

} // let Flags = [CC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// Frontend Options
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, NoDriverOption] in {

// This isn't normally used, it is just here so we can parse a
// CompilerInvocation out of a driver-derived argument vector.
def cc1 : Flag<["-"], "cc1">;
def cc1as : Flag<["-"], "cc1as">;

def ast_merge : Separate<["-"], "ast-merge">,
  MetaVarName<"<ast file>">,
  HelpText<"Merge the given AST file into the translation unit being compiled.">,
  MarshallingInfoStringVector<FrontendOpts<"ASTMergeFiles">>;
def aux_target_cpu : Separate<["-"], "aux-target-cpu">,
  HelpText<"Target a specific auxiliary cpu type">;
def aux_target_feature : Separate<["-"], "aux-target-feature">,
  HelpText<"Target specific auxiliary attributes">;
def aux_triple : Separate<["-"], "aux-triple">,
  HelpText<"Auxiliary target triple.">,
  MarshallingInfoString<FrontendOpts<"AuxTriple">>;
def code_completion_at : Separate<["-"], "code-completion-at">,
  MetaVarName<"<file>:<line>:<column>">,
  HelpText<"Dump code-completion information at a location">;
def remap_file : Separate<["-"], "remap-file">,
  MetaVarName<"<from>;<to>">,
  HelpText<"Replace the contents of the <from> file with the contents of the <to> file">;
def code_completion_at_EQ : Joined<["-"], "code-completion-at=">,
  Alias<code_completion_at>;
def code_completion_macros : Flag<["-"], "code-completion-macros">,
  HelpText<"Include macros in code-completion results">,
  MarshallingInfoFlag<FrontendOpts<"CodeCompleteOpts.IncludeMacros">>;
def code_completion_patterns : Flag<["-"], "code-completion-patterns">,
  HelpText<"Include code patterns in code-completion results">,
  MarshallingInfoFlag<FrontendOpts<"CodeCompleteOpts.IncludeCodePatterns">>;
def no_code_completion_globals : Flag<["-"], "no-code-completion-globals">,
  HelpText<"Do not include global declarations in code-completion results.">,
  MarshallingInfoNegativeFlag<FrontendOpts<"CodeCompleteOpts.IncludeGlobals">>;
def no_code_completion_ns_level_decls : Flag<["-"], "no-code-completion-ns-level-decls">,
  HelpText<"Do not include declarations inside namespaces (incl. global namespace) in the code-completion results.">,
  MarshallingInfoNegativeFlag<FrontendOpts<"CodeCompleteOpts.IncludeNamespaceLevelDecls">>;
def code_completion_brief_comments : Flag<["-"], "code-completion-brief-comments">,
  HelpText<"Include brief documentation comments in code-completion results.">,
  MarshallingInfoFlag<FrontendOpts<"CodeCompleteOpts.IncludeBriefComments">>;
def code_completion_with_fixits : Flag<["-"], "code-completion-with-fixits">,
  HelpText<"Include code completion results which require small fix-its.">,
  MarshallingInfoFlag<FrontendOpts<"CodeCompleteOpts.IncludeFixIts">>;
def disable_free : Flag<["-"], "disable-free">,
  HelpText<"Disable freeing of memory on exit">,
  MarshallingInfoFlag<FrontendOpts<"DisableFree">>;
defm clear_ast_before_backend : BoolOption<"",
  "clear-ast-before-backend",
  CodeGenOpts<"ClearASTBeforeBackend">,
  DefaultFalse,
  PosFlag<SetTrue, [], "Clear">,
  NegFlag<SetFalse, [], "Don't clear">,
  BothFlags<[], " the Clang AST before running backend code generation">>;
defm enable_noundef_analysis : BoolOption<"",
  "enable-noundef-analysis",
  CodeGenOpts<"EnableNoundefAttrs">,
  DefaultTrue,
  PosFlag<SetTrue, [], "Enable">,
  NegFlag<SetFalse, [], "Disable">,
  BothFlags<[], " analyzing function argument and return types for mandatory definedness">>;
defm opaque_pointers : BoolOption<"",
  "opaque-pointers",
  CodeGenOpts<"OpaquePointers">,
  DefaultTrue,
  PosFlag<SetTrue, [], "Enable">,
  NegFlag<SetFalse, [], "Disable">,
  BothFlags<[], " opaque pointers">>;
def discard_value_names : Flag<["-"], "discard-value-names">,
  HelpText<"Discard value names in LLVM IR">,
  MarshallingInfoFlag<CodeGenOpts<"DiscardValueNames">>;
def plugin_arg : JoinedAndSeparate<["-"], "plugin-arg-">,
    MetaVarName<"<name> <arg>">,
    HelpText<"Pass <arg> to plugin <name>">;
def add_plugin : Separate<["-"], "add-plugin">, MetaVarName<"<name>">,
  HelpText<"Use the named plugin action in addition to the default action">,
  MarshallingInfoStringVector<FrontendOpts<"AddPluginActions">>;
def ast_dump_filter : Separate<["-"], "ast-dump-filter">,
  MetaVarName<"<dump_filter>">,
  HelpText<"Use with -ast-dump or -ast-print to dump/print only AST declaration"
           " nodes having a certain substring in a qualified name. Use"
           " -ast-list to list all filterable declaration node names.">,
  MarshallingInfoString<FrontendOpts<"ASTDumpFilter">>;
def ast_dump_filter_EQ : Joined<["-"], "ast-dump-filter=">,
  Alias<ast_dump_filter>;
def fno_modules_global_index : Flag<["-"], "fno-modules-global-index">,
  HelpText<"Do not automatically generate or update the global module index">,
  MarshallingInfoNegativeFlag<FrontendOpts<"UseGlobalModuleIndex">>;
def fno_modules_error_recovery : Flag<["-"], "fno-modules-error-recovery">,
  HelpText<"Do not automatically import modules for error recovery">,
  MarshallingInfoNegativeFlag<LangOpts<"ModulesErrorRecovery">>;
def fmodule_map_file_home_is_cwd : Flag<["-"], "fmodule-map-file-home-is-cwd">,
  HelpText<"Use the current working directory as the home directory of "
           "module maps specified by -fmodule-map-file=<FILE>">,
  MarshallingInfoFlag<HeaderSearchOpts<"ModuleMapFileHomeIsCwd">>;
def fmodule_file_home_is_cwd : Flag<["-"], "fmodule-file-home-is-cwd">,
  HelpText<"Use the current working directory as the base directory of "
           "compiled module files.">,
  MarshallingInfoFlag<HeaderSearchOpts<"ModuleFileHomeIsCwd">>;
def fmodule_feature : Separate<["-"], "fmodule-feature">,
  MetaVarName<"<feature>">,
  HelpText<"Enable <feature> in module map requires declarations">,
  MarshallingInfoStringVector<LangOpts<"ModuleFeatures">>;
def fmodules_embed_file_EQ : Joined<["-"], "fmodules-embed-file=">,
  MetaVarName<"<file>">,
  HelpText<"Embed the contents of the specified file into the module file "
           "being compiled.">,
  MarshallingInfoStringVector<FrontendOpts<"ModulesEmbedFiles">>;
def fmodules_embed_all_files : Joined<["-"], "fmodules-embed-all-files">,
  HelpText<"Embed the contents of all files read by this compilation into "
           "the produced module file.">,
  MarshallingInfoFlag<FrontendOpts<"ModulesEmbedAllFiles">>;
defm fimplicit_modules_use_lock : BoolOption<"f", "implicit-modules-use-lock",
  FrontendOpts<"BuildingImplicitModuleUsesLock">, DefaultTrue,
  NegFlag<SetFalse>,
  PosFlag<SetTrue, [],
          "Use filesystem locks for implicit modules builds to avoid "
          "duplicating work in competing clang invocations.">>;
// FIXME: We only need this in C++ modules / Modules TS if we might textually
// enter a different module (eg, when building a header unit).
def fmodules_local_submodule_visibility :
  Flag<["-"], "fmodules-local-submodule-visibility">,
  HelpText<"Enforce name visibility rules across submodules of the same "
           "top-level module.">,
  MarshallingInfoFlag<LangOpts<"ModulesLocalVisibility">>,
  ImpliedByAnyOf<[fmodules_ts.KeyPath, fcxx_modules.KeyPath]>;
def fmodules_codegen :
  Flag<["-"], "fmodules-codegen">,
  HelpText<"Generate code for uses of this module that assumes an explicit "
           "object file will be built for the module">,
  MarshallingInfoFlag<LangOpts<"ModulesCodegen">>;
def fmodules_debuginfo :
  Flag<["-"], "fmodules-debuginfo">,
  HelpText<"Generate debug info for types in an object file built from this "
           "module and do not generate them elsewhere">,
  MarshallingInfoFlag<LangOpts<"ModulesDebugInfo">>;
def fmodule_format_EQ : Joined<["-"], "fmodule-format=">,
  HelpText<"Select the container format for clang modules and PCH. "
           "Supported options are 'raw' and 'obj'.">,
  MarshallingInfoString<HeaderSearchOpts<"ModuleFormat">, [{"raw"}]>;
def ftest_module_file_extension_EQ :
  Joined<["-"], "ftest-module-file-extension=">,
  HelpText<"introduce a module file extension for testing purposes. "
           "The argument is parsed as blockname:major:minor:hashed:user info">;
def fconcepts_ts : Flag<["-"], "fconcepts-ts">,
  HelpText<"Enable C++ Extensions for Concepts. (deprecated - use -std=c++2a)">;

defm recovery_ast : BoolOption<"f", "recovery-ast",
  LangOpts<"RecoveryAST">, DefaultTrue,
  NegFlag<SetFalse>, PosFlag<SetTrue, [], "Preserve expressions in AST rather "
                              "than dropping them when encountering semantic errors">>;
defm recovery_ast_type : BoolOption<"f", "recovery-ast-type",
  LangOpts<"RecoveryASTType">, DefaultTrue,
  NegFlag<SetFalse>, PosFlag<SetTrue, [], "Preserve the type for recovery "
                              "expressions when possible">>;

let Group = Action_Group in {

def Eonly : Flag<["-"], "Eonly">,
  HelpText<"Just run preprocessor, no output (for timings)">;
def dump_raw_tokens : Flag<["-"], "dump-raw-tokens">,
  HelpText<"Lex file in raw mode and dump raw tokens">;
def analyze : Flag<["-"], "analyze">,
  HelpText<"Run static analysis engine">;
def dump_tokens : Flag<["-"], "dump-tokens">,
  HelpText<"Run preprocessor, dump internal rep of tokens">;
def fixit : Flag<["-"], "fixit">,
  HelpText<"Apply fix-it advice to the input source">;
def fixit_EQ : Joined<["-"], "fixit=">,
  HelpText<"Apply fix-it advice creating a file with the given suffix">;
def print_preamble : Flag<["-"], "print-preamble">,
  HelpText<"Print the \"preamble\" of a file, which is a candidate for implicit"
           " precompiled headers.">;
def emit_html : Flag<["-"], "emit-html">,
  HelpText<"Output input source as HTML">;
def ast_print : Flag<["-"], "ast-print">,
  HelpText<"Build ASTs and then pretty-print them">;
def ast_list : Flag<["-"], "ast-list">,
  HelpText<"Build ASTs and print the list of declaration node qualified names">;
def ast_dump : Flag<["-"], "ast-dump">,
  HelpText<"Build ASTs and then debug dump them">;
def ast_dump_EQ : Joined<["-"], "ast-dump=">,
  HelpText<"Build ASTs and then debug dump them in the specified format. "
           "Supported formats include: default, json">;
def ast_dump_all : Flag<["-"], "ast-dump-all">,
  HelpText<"Build ASTs and then debug dump them, forcing deserialization">;
def ast_dump_all_EQ : Joined<["-"], "ast-dump-all=">,
  HelpText<"Build ASTs and then debug dump them in the specified format, "
           "forcing deserialization. Supported formats include: default, json">;
def ast_dump_decl_types : Flag<["-"], "ast-dump-decl-types">,
  HelpText<"Include declaration types in AST dumps">,
  MarshallingInfoFlag<FrontendOpts<"ASTDumpDeclTypes">>;
def templight_dump : Flag<["-"], "templight-dump">,
  HelpText<"Dump templight information to stdout">;
def ast_dump_lookups : Flag<["-"], "ast-dump-lookups">,
  HelpText<"Build ASTs and then debug dump their name lookup tables">,
  MarshallingInfoFlag<FrontendOpts<"ASTDumpLookups">>;
def ast_view : Flag<["-"], "ast-view">,
  HelpText<"Build ASTs and view them with GraphViz">;
def emit_module : Flag<["-"], "emit-module">,
  HelpText<"Generate pre-compiled module file from a module map">;
def emit_module_interface : Flag<["-"], "emit-module-interface">,
  HelpText<"Generate pre-compiled module file from a C++ module interface">;
def emit_header_module : Flag<["-"], "emit-header-module">,
  HelpText<"Generate pre-compiled module file from a set of header files">;
def emit_header_unit : Flag<["-"], "emit-header-unit">,
  HelpText<"Generate C++20 header units from header files">;
def emit_pch : Flag<["-"], "emit-pch">,
  HelpText<"Generate pre-compiled header file">;
def emit_llvm_only : Flag<["-"], "emit-llvm-only">,
  HelpText<"Build ASTs and convert to LLVM, discarding output">;
def emit_codegen_only : Flag<["-"], "emit-codegen-only">,
  HelpText<"Generate machine code, but discard output">;
def rewrite_test : Flag<["-"], "rewrite-test">,
  HelpText<"Rewriter playground">;
def rewrite_macros : Flag<["-"], "rewrite-macros">,
  HelpText<"Expand macros without full preprocessing">;
def migrate : Flag<["-"], "migrate">,
  HelpText<"Migrate source code">;
def compiler_options_dump : Flag<["-"], "compiler-options-dump">,
  HelpText<"Dump the compiler configuration options">;
def print_dependency_directives_minimized_source : Flag<["-"],
  "print-dependency-directives-minimized-source">,
  HelpText<"Print the output of the dependency directives source minimizer">;
}

defm emit_llvm_uselists : BoolOption<"", "emit-llvm-uselists",
  CodeGenOpts<"EmitLLVMUseLists">, DefaultFalse,
  PosFlag<SetTrue, [], "Preserve">,
  NegFlag<SetFalse, [], "Don't preserve">,
  BothFlags<[], " order of LLVM use-lists when serializing">>;

def mt_migrate_directory : Separate<["-"], "mt-migrate-directory">,
  HelpText<"Directory for temporary files produced during ARC or ObjC migration">,
  MarshallingInfoString<FrontendOpts<"MTMigrateDir">>;

def arcmt_action_EQ : Joined<["-"], "arcmt-action=">, Flags<[CC1Option, NoDriverOption]>,
  HelpText<"The ARC migration action to take">,
  Values<"check,modify,migrate">,
  NormalizedValuesScope<"FrontendOptions">,
  NormalizedValues<["ARCMT_Check", "ARCMT_Modify", "ARCMT_Migrate"]>,
  MarshallingInfoEnum<FrontendOpts<"ARCMTAction">, "ARCMT_None">;

def opt_record_file : Separate<["-"], "opt-record-file">,
  HelpText<"File name to use for YAML optimization record output">,
  MarshallingInfoString<CodeGenOpts<"OptRecordFile">>;
def opt_record_passes : Separate<["-"], "opt-record-passes">,
  HelpText<"Only record remark information for passes whose names match the given regular expression">;
def opt_record_format : Separate<["-"], "opt-record-format">,
  HelpText<"The format used for serializing remarks (default: YAML)">;

def print_stats : Flag<["-"], "print-stats">,
  HelpText<"Print performance metrics and statistics">,
  MarshallingInfoFlag<FrontendOpts<"ShowStats">>;
def stats_file : Joined<["-"], "stats-file=">,
  HelpText<"Filename to write statistics to">,
  MarshallingInfoString<FrontendOpts<"StatsFile">>;
def fdump_record_layouts_simple : Flag<["-"], "fdump-record-layouts-simple">,
  HelpText<"Dump record layout information in a simple form used for testing">,
  MarshallingInfoFlag<LangOpts<"DumpRecordLayoutsSimple">>;
def fdump_record_layouts_canonical : Flag<["-"], "fdump-record-layouts-canonical">,
  HelpText<"Dump record layout information with canonical field types">,
  MarshallingInfoFlag<LangOpts<"DumpRecordLayoutsCanonical">>;
def fdump_record_layouts_complete : Flag<["-"], "fdump-record-layouts-complete">,
  HelpText<"Dump record layout information for all complete types">,
  MarshallingInfoFlag<LangOpts<"DumpRecordLayoutsComplete">>;
def fdump_record_layouts : Flag<["-"], "fdump-record-layouts">,
  HelpText<"Dump record layout information">,
  MarshallingInfoFlag<LangOpts<"DumpRecordLayouts">>,
  ImpliedByAnyOf<[fdump_record_layouts_simple.KeyPath, fdump_record_layouts_complete.KeyPath, fdump_record_layouts_canonical.KeyPath]>;
def fix_what_you_can : Flag<["-"], "fix-what-you-can">,
  HelpText<"Apply fix-it advice even in the presence of unfixable errors">,
  MarshallingInfoFlag<FrontendOpts<"FixWhatYouCan">>;
def fix_only_warnings : Flag<["-"], "fix-only-warnings">,
  HelpText<"Apply fix-it advice only for warnings, not errors">,
  MarshallingInfoFlag<FrontendOpts<"FixOnlyWarnings">>;
def fixit_recompile : Flag<["-"], "fixit-recompile">,
  HelpText<"Apply fix-it changes and recompile">,
  MarshallingInfoFlag<FrontendOpts<"FixAndRecompile">>;
def fixit_to_temp : Flag<["-"], "fixit-to-temporary">,
  HelpText<"Apply fix-it changes to temporary files">,
  MarshallingInfoFlag<FrontendOpts<"FixToTemporaries">>;

def foverride_record_layout_EQ : Joined<["-"], "foverride-record-layout=">,
  HelpText<"Override record layouts with those in the given file">,
  MarshallingInfoString<FrontendOpts<"OverrideRecordLayoutsFile">>;
def pch_through_header_EQ : Joined<["-"], "pch-through-header=">,
  HelpText<"Stop PCH generation after including this file.  When using a PCH, "
           "skip tokens until after this file is included.">,
  MarshallingInfoString<PreprocessorOpts<"PCHThroughHeader">>;
def pch_through_hdrstop_create : Flag<["-"], "pch-through-hdrstop-create">,
  HelpText<"When creating a PCH, stop PCH generation after #pragma hdrstop.">,
  MarshallingInfoFlag<PreprocessorOpts<"PCHWithHdrStopCreate">>;
def pch_through_hdrstop_use : Flag<["-"], "pch-through-hdrstop-use">,
  HelpText<"When using a PCH, skip tokens until after a #pragma hdrstop.">;
def fno_pch_timestamp : Flag<["-"], "fno-pch-timestamp">,
  HelpText<"Disable inclusion of timestamp in precompiled headers">,
  MarshallingInfoNegativeFlag<FrontendOpts<"IncludeTimestamps">>;
def building_pch_with_obj : Flag<["-"], "building-pch-with-obj">,
  HelpText<"This compilation is part of building a PCH with corresponding object file.">,
  MarshallingInfoFlag<LangOpts<"BuildingPCHWithObjectFile">>;

def aligned_alloc_unavailable : Flag<["-"], "faligned-alloc-unavailable">,
  HelpText<"Aligned allocation/deallocation functions are unavailable">,
  MarshallingInfoFlag<LangOpts<"AlignedAllocationUnavailable">>,
  ShouldParseIf<faligned_allocation.KeyPath>;

} // let Flags = [CC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// Language Options
//===----------------------------------------------------------------------===//

def version : Flag<["-"], "version">,
  HelpText<"Print the compiler version">,
  Flags<[CC1Option, CC1AsOption, FC1Option, NoDriverOption]>,
  MarshallingInfoFlag<FrontendOpts<"ShowVersion">>;

def main_file_name : Separate<["-"], "main-file-name">,
  HelpText<"Main file name to use for debug info and source if missing">,
  Flags<[CC1Option, CC1AsOption, NoDriverOption]>,
  MarshallingInfoString<CodeGenOpts<"MainFileName">>;
def split_dwarf_output : Separate<["-"], "split-dwarf-output">,
  HelpText<"File name to use for split dwarf debug info output">,
  Flags<[CC1Option, CC1AsOption, NoDriverOption]>,
  MarshallingInfoString<CodeGenOpts<"SplitDwarfOutput">>;

let Flags = [CC1Option, NoDriverOption] in {

def fblocks_runtime_optional : Flag<["-"], "fblocks-runtime-optional">,
  HelpText<"Weakly link in the blocks runtime">,
  MarshallingInfoFlag<LangOpts<"BlocksRuntimeOptional">>;
def fexternc_nounwind : Flag<["-"], "fexternc-nounwind">,
  HelpText<"Assume all functions with C linkage do not unwind">,
  MarshallingInfoFlag<LangOpts<"ExternCNoUnwind">>;
def split_dwarf_file : Separate<["-"], "split-dwarf-file">,
  HelpText<"Name of the split dwarf debug info file to encode in the object file">,
  MarshallingInfoString<CodeGenOpts<"SplitDwarfFile">>;
def fno_wchar : Flag<["-"], "fno-wchar">,
  HelpText<"Disable C++ builtin type wchar_t">,
  MarshallingInfoNegativeFlag<LangOpts<"WChar">, cplusplus.KeyPath>,
  ShouldParseIf<cplusplus.KeyPath>;
def fconstant_string_class : Separate<["-"], "fconstant-string-class">,
  MetaVarName<"<class name>">,
  HelpText<"Specify the class to use for constant Objective-C string objects.">,
  MarshallingInfoString<LangOpts<"ObjCConstantStringClass">>;
def fobjc_arc_cxxlib_EQ : Joined<["-"], "fobjc-arc-cxxlib=">,
  HelpText<"Objective-C++ Automatic Reference Counting standard library kind">,
  Values<"libc++,libstdc++,none">,
  NormalizedValues<["ARCXX_libcxx", "ARCXX_libstdcxx", "ARCXX_nolib"]>,
  MarshallingInfoEnum<PreprocessorOpts<"ObjCXXARCStandardLibrary">, "ARCXX_nolib">;
def fobjc_runtime_has_weak : Flag<["-"], "fobjc-runtime-has-weak">,
  HelpText<"The target Objective-C runtime supports ARC weak operations">;
def fobjc_dispatch_method_EQ : Joined<["-"], "fobjc-dispatch-method=">,
  HelpText<"Objective-C dispatch method to use">,
  Values<"legacy,non-legacy,mixed">,
  NormalizedValuesScope<"CodeGenOptions">, NormalizedValues<["Legacy", "NonLegacy", "Mixed"]>,
  MarshallingInfoEnum<CodeGenOpts<"ObjCDispatchMethod">, "Legacy">;
def disable_objc_default_synthesize_properties : Flag<["-"], "disable-objc-default-synthesize-properties">,
  HelpText<"disable the default synthesis of Objective-C properties">,
  MarshallingInfoNegativeFlag<LangOpts<"ObjCDefaultSynthProperties">>;
def fencode_extended_block_signature : Flag<["-"], "fencode-extended-block-signature">,
  HelpText<"enable extended encoding of block type signature">,
  MarshallingInfoFlag<LangOpts<"EncodeExtendedBlockSig">>;
def function_alignment : Separate<["-"], "function-alignment">,
    HelpText<"default alignment for functions">,
    MarshallingInfoInt<LangOpts<"FunctionAlignment">>;
def pic_level : Separate<["-"], "pic-level">,
  HelpText<"Value for __PIC__">,
  MarshallingInfoInt<LangOpts<"PICLevel">>;
def pic_is_pie : Flag<["-"], "pic-is-pie">,
  HelpText<"File is for a position independent executable">,
  MarshallingInfoFlag<LangOpts<"PIE">>;
def fhalf_no_semantic_interposition : Flag<["-"], "fhalf-no-semantic-interposition">,
  HelpText<"Like -fno-semantic-interposition but don't use local aliases">,
  MarshallingInfoFlag<LangOpts<"HalfNoSemanticInterposition">>;
def fno_validate_pch : Flag<["-"], "fno-validate-pch">,
  HelpText<"Disable validation of precompiled headers">,
  MarshallingInfoFlag<PreprocessorOpts<"DisablePCHOrModuleValidation">, "DisableValidationForModuleKind::None">,
  Normalizer<"makeFlagToValueNormalizer(DisableValidationForModuleKind::All)">;
def fallow_pcm_with_errors : Flag<["-"], "fallow-pcm-with-compiler-errors">,
  HelpText<"Accept a PCM file that was created with compiler errors">,
  MarshallingInfoFlag<FrontendOpts<"AllowPCMWithCompilerErrors">>;
def fallow_pch_with_errors : Flag<["-"], "fallow-pch-with-compiler-errors">,
  HelpText<"Accept a PCH file that was created with compiler errors">,
  MarshallingInfoFlag<PreprocessorOpts<"AllowPCHWithCompilerErrors">>,
  ImpliedByAnyOf<[fallow_pcm_with_errors.KeyPath]>;
def fallow_pch_with_different_modules_cache_path :
  Flag<["-"], "fallow-pch-with-different-modules-cache-path">,
  HelpText<"Accept a PCH file that was created with a different modules cache path">,
  MarshallingInfoFlag<PreprocessorOpts<"AllowPCHWithDifferentModulesCachePath">>;
def dump_deserialized_pch_decls : Flag<["-"], "dump-deserialized-decls">,
  HelpText<"Dump declarations that are deserialized from PCH, for testing">,
  MarshallingInfoFlag<PreprocessorOpts<"DumpDeserializedPCHDecls">>;
def error_on_deserialized_pch_decl : Separate<["-"], "error-on-deserialized-decl">,
  HelpText<"Emit error if a specific declaration is deserialized from PCH, for testing">;
def error_on_deserialized_pch_decl_EQ : Joined<["-"], "error-on-deserialized-decl=">,
  Alias<error_on_deserialized_pch_decl>;
def static_define : Flag<["-"], "static-define">,
  HelpText<"Should __STATIC__ be defined">,
  MarshallingInfoFlag<LangOpts<"Static">>;
def stack_protector : Separate<["-"], "stack-protector">,
  HelpText<"Enable stack protectors">,
  Values<"0,1,2,3">,
  NormalizedValuesScope<"LangOptions">,
  NormalizedValues<["SSPOff", "SSPOn", "SSPStrong", "SSPReq"]>,
  MarshallingInfoEnum<LangOpts<"StackProtector">, "SSPOff">;
def stack_protector_buffer_size : Separate<["-"], "stack-protector-buffer-size">,
  HelpText<"Lower bound for a buffer to be considered for stack protection">,
  MarshallingInfoInt<CodeGenOpts<"SSPBufferSize">, "8">;
def fvisibility : Separate<["-"], "fvisibility">,
  HelpText<"Default type and symbol visibility">,
  MarshallingInfoVisibility<LangOpts<"ValueVisibilityMode">, "DefaultVisibility">;
def ftype_visibility : Separate<["-"], "ftype-visibility">,
  HelpText<"Default type visibility">,
  MarshallingInfoVisibility<LangOpts<"TypeVisibilityMode">, fvisibility.KeyPath>;
def fapply_global_visibility_to_externs : Flag<["-"], "fapply-global-visibility-to-externs">,
  HelpText<"Apply global symbol visibility to external declarations without an explicit visibility">,
  MarshallingInfoFlag<LangOpts<"SetVisibilityForExternDecls">>;
def ftemplate_depth : Separate<["-"], "ftemplate-depth">,
  HelpText<"Maximum depth of recursive template instantiation">,
  MarshallingInfoInt<LangOpts<"InstantiationDepth">, "1024">;
def foperator_arrow_depth : Separate<["-"], "foperator-arrow-depth">,
  HelpText<"Maximum number of 'operator->'s to call for a member access">,
  MarshallingInfoInt<LangOpts<"ArrowDepth">, "256">;
def fconstexpr_depth : Separate<["-"], "fconstexpr-depth">,
  HelpText<"Maximum depth of recursive constexpr function calls">,
  MarshallingInfoInt<LangOpts<"ConstexprCallDepth">, "512">;
def fconstexpr_steps : Separate<["-"], "fconstexpr-steps">,
  HelpText<"Maximum number of steps in constexpr function evaluation">,
  MarshallingInfoInt<LangOpts<"ConstexprStepLimit">, "1048576">;
def fbracket_depth : Separate<["-"], "fbracket-depth">,
  HelpText<"Maximum nesting level for parentheses, brackets, and braces">,
  MarshallingInfoInt<LangOpts<"BracketDepth">, "256">;
defm const_strings : BoolOption<"f", "const-strings",
  LangOpts<"ConstStrings">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Use">, NegFlag<SetFalse, [], "Don't use">,
  BothFlags<[], " a const qualified type for string literals in C and ObjC">>;
def fno_bitfield_type_align : Flag<["-"], "fno-bitfield-type-align">,
  HelpText<"Ignore bit-field types when aligning structures">,
  MarshallingInfoFlag<LangOpts<"NoBitFieldTypeAlign">>;
def ffake_address_space_map : Flag<["-"], "ffake-address-space-map">,
  HelpText<"Use a fake address space map; OpenCL testing purposes only">,
  MarshallingInfoFlag<LangOpts<"FakeAddressSpaceMap">>;
def faddress_space_map_mangling_EQ : Joined<["-"], "faddress-space-map-mangling=">,
  HelpText<"Set the mode for address space map based mangling; OpenCL testing purposes only">,
  Values<"target,no,yes">,
  NormalizedValuesScope<"LangOptions">,
  NormalizedValues<["ASMM_Target", "ASMM_Off", "ASMM_On"]>,
  MarshallingInfoEnum<LangOpts<"AddressSpaceMapMangling">, "ASMM_Target">;
def funknown_anytype : Flag<["-"], "funknown-anytype">,
  HelpText<"Enable parser support for the __unknown_anytype type; for testing purposes only">,
  MarshallingInfoFlag<LangOpts<"ParseUnknownAnytype">>;
def fdebugger_support : Flag<["-"], "fdebugger-support">,
  HelpText<"Enable special debugger support behavior">,
  MarshallingInfoFlag<LangOpts<"DebuggerSupport">>;
def fdebugger_cast_result_to_id : Flag<["-"], "fdebugger-cast-result-to-id">,
  HelpText<"Enable casting unknown expression results to id">,
  MarshallingInfoFlag<LangOpts<"DebuggerCastResultToId">>;
def fdebugger_objc_literal : Flag<["-"], "fdebugger-objc-literal">,
  HelpText<"Enable special debugger support for Objective-C subscripting and literals">,
  MarshallingInfoFlag<LangOpts<"DebuggerObjCLiteral">>;
defm deprecated_macro : BoolOption<"f", "deprecated-macro",
  LangOpts<"Deprecated">, DefaultFalse,
  PosFlag<SetTrue, [], "Defines">, NegFlag<SetFalse, [], "Undefines">,
  BothFlags<[], " the __DEPRECATED macro">>;
def fobjc_subscripting_legacy_runtime : Flag<["-"], "fobjc-subscripting-legacy-runtime">,
  HelpText<"Allow Objective-C array and dictionary subscripting in legacy runtime">;
// TODO: Enforce values valid for MSVtorDispMode.
def vtordisp_mode_EQ : Joined<["-"], "vtordisp-mode=">,
  HelpText<"Control vtordisp placement on win32 targets">,
  MarshallingInfoInt<LangOpts<"VtorDispMode">, "1">;
def fnative_half_type: Flag<["-"], "fnative-half-type">,
  HelpText<"Use the native half type for __fp16 instead of promoting to float">,
  MarshallingInfoFlag<LangOpts<"NativeHalfType">>,
  ImpliedByAnyOf<[open_cl.KeyPath, render_script.KeyPath]>;
def fnative_half_arguments_and_returns : Flag<["-"], "fnative-half-arguments-and-returns">,
  HelpText<"Use the native __fp16 type for arguments and returns (and skip ABI-specific lowering)">,
  MarshallingInfoFlag<LangOpts<"NativeHalfArgsAndReturns">>,
  ImpliedByAnyOf<[open_cl.KeyPath, render_script.KeyPath]>;
def fallow_half_arguments_and_returns : Flag<["-"], "fallow-half-arguments-and-returns">,
  HelpText<"Allow function arguments and returns of type half">,
  MarshallingInfoFlag<LangOpts<"HalfArgsAndReturns">>,
  ImpliedByAnyOf<[fnative_half_arguments_and_returns.KeyPath]>;
def fdefault_calling_conv_EQ : Joined<["-"], "fdefault-calling-conv=">,
  HelpText<"Set default calling convention">,
  Values<"cdecl,fastcall,stdcall,vectorcall,regcall">,
  NormalizedValuesScope<"LangOptions">,
  NormalizedValues<["DCC_CDecl", "DCC_FastCall", "DCC_StdCall", "DCC_VectorCall", "DCC_RegCall"]>,
  MarshallingInfoEnum<LangOpts<"DefaultCallingConv">, "DCC_None">;

// These options cannot be marshalled, because they are used to set up the LangOptions defaults.
def finclude_default_header : Flag<["-"], "finclude-default-header">,
  HelpText<"Include default header file for OpenCL and HLSL">;
def fdeclare_opencl_builtins : Flag<["-"], "fdeclare-opencl-builtins">,
  HelpText<"Add OpenCL builtin function declarations (experimental)">;

def fpreserve_vec3_type : Flag<["-"], "fpreserve-vec3-type">,
  HelpText<"Preserve 3-component vector type">,
  MarshallingInfoFlag<CodeGenOpts<"PreserveVec3Type">>;
def fwchar_type_EQ : Joined<["-"], "fwchar-type=">,
  HelpText<"Select underlying type for wchar_t">,
  Values<"char,short,int">,
  NormalizedValues<["1", "2", "4"]>,
  MarshallingInfoEnum<LangOpts<"WCharSize">, "0">;
defm signed_wchar : BoolOption<"f", "signed-wchar",
  LangOpts<"WCharIsSigned">, DefaultTrue,
  NegFlag<SetFalse, [CC1Option], "Use an unsigned">, PosFlag<SetTrue, [], "Use a signed">,
  BothFlags<[], " type for wchar_t">>;
def fcompatibility_qualified_id_block_param_type_checking : Flag<["-"], "fcompatibility-qualified-id-block-type-checking">,
  HelpText<"Allow using blocks with parameters of more specific type than "
           "the type system guarantees when a parameter is qualified id">,
  MarshallingInfoFlag<LangOpts<"CompatibilityQualifiedIdBlockParamTypeChecking">>;
def fpass_by_value_is_noalias: Flag<["-"], "fpass-by-value-is-noalias">,
  HelpText<"Allows assuming by-value parameters do not alias any other value. "
           "Has no effect on non-trivially-copyable classes in C++.">, Group<f_Group>,
  MarshallingInfoFlag<CodeGenOpts<"PassByValueIsNoAlias">>;

// FIXME: Remove these entirely once functionality/tests have been excised.
def fobjc_gc_only : Flag<["-"], "fobjc-gc-only">, Group<f_Group>,
  HelpText<"Use GC exclusively for Objective-C related memory management">;
def fobjc_gc : Flag<["-"], "fobjc-gc">, Group<f_Group>,
  HelpText<"Enable Objective-C garbage collection">;

def fexperimental_max_bitint_width_EQ:
  Joined<["-"], "fexperimental-max-bitint-width=">, Group<f_Group>,
  MetaVarName<"<N>">,
  HelpText<"Set the maximum bitwidth for _BitInt (this option is expected to be removed in the future)">,
  MarshallingInfoInt<LangOpts<"MaxBitIntWidth">>;

} // let Flags = [CC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// Header Search Options
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, NoDriverOption] in {

def nostdsysteminc : Flag<["-"], "nostdsysteminc">,
  HelpText<"Disable standard system #include directories">,
  MarshallingInfoNegativeFlag<HeaderSearchOpts<"UseStandardSystemIncludes">>;
def fdisable_module_hash : Flag<["-"], "fdisable-module-hash">,
  HelpText<"Disable the module hash">,
  MarshallingInfoFlag<HeaderSearchOpts<"DisableModuleHash">>;
def fmodules_hash_content : Flag<["-"], "fmodules-hash-content">,
  HelpText<"Enable hashing the content of a module file">,
  MarshallingInfoFlag<HeaderSearchOpts<"ModulesHashContent">>;
def fmodules_strict_context_hash : Flag<["-"], "fmodules-strict-context-hash">,
  HelpText<"Enable hashing of all compiler options that could impact the "
           "semantics of a module in an implicit build">,
  MarshallingInfoFlag<HeaderSearchOpts<"ModulesStrictContextHash">>;
def c_isystem : JoinedOrSeparate<["-"], "c-isystem">, MetaVarName<"<directory>">,
  HelpText<"Add directory to the C SYSTEM include search path">;
def objc_isystem : JoinedOrSeparate<["-"], "objc-isystem">,
  MetaVarName<"<directory>">,
  HelpText<"Add directory to the ObjC SYSTEM include search path">;
def objcxx_isystem : JoinedOrSeparate<["-"], "objcxx-isystem">,
  MetaVarName<"<directory>">,
  HelpText<"Add directory to the ObjC++ SYSTEM include search path">;
def internal_isystem : JoinedOrSeparate<["-"], "internal-isystem">,
  MetaVarName<"<directory>">,
  HelpText<"Add directory to the internal system include search path; these "
           "are assumed to not be user-provided and are used to model system "
           "and standard headers' paths.">;
def internal_externc_isystem : JoinedOrSeparate<["-"], "internal-externc-isystem">,
  MetaVarName<"<directory>">,
  HelpText<"Add directory to the internal system include search path with "
           "implicit extern \"C\" semantics; these are assumed to not be "
           "user-provided and are used to model system and standard headers' "
           "paths.">;

} // let Flags = [CC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// Preprocessor Options
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, NoDriverOption] in {

def chain_include : Separate<["-"], "chain-include">, MetaVarName<"<file>">,
  HelpText<"Include and chain a header file after turning it into PCH">;
def preamble_bytes_EQ : Joined<["-"], "preamble-bytes=">,
  HelpText<"Assume that the precompiled header is a precompiled preamble "
           "covering the first N bytes of the main file">;
def detailed_preprocessing_record : Flag<["-"], "detailed-preprocessing-record">,
  HelpText<"include a detailed record of preprocessing actions">,
  MarshallingInfoFlag<PreprocessorOpts<"DetailedRecord">>;
def setup_static_analyzer : Flag<["-"], "setup-static-analyzer">,
  HelpText<"Set up preprocessor for static analyzer (done automatically when static analyzer is run).">,
  MarshallingInfoFlag<PreprocessorOpts<"SetUpStaticAnalyzer">>;
def disable_pragma_debug_crash : Flag<["-"], "disable-pragma-debug-crash">,
  HelpText<"Disable any #pragma clang __debug that can lead to crashing behavior. This is meant for testing.">,
  MarshallingInfoFlag<PreprocessorOpts<"DisablePragmaDebugCrash">>;

} // let Flags = [CC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// CUDA Options
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, NoDriverOption] in {

def fcuda_is_device : Flag<["-"], "fcuda-is-device">,
  HelpText<"Generate code for CUDA device">,
  MarshallingInfoFlag<LangOpts<"CUDAIsDevice">>;
def fcuda_include_gpubinary : Separate<["-"], "fcuda-include-gpubinary">,
  HelpText<"Incorporate CUDA device-side binary into host object file.">,
  MarshallingInfoString<CodeGenOpts<"CudaGpuBinaryFileName">>;
def fcuda_allow_variadic_functions : Flag<["-"], "fcuda-allow-variadic-functions">,
  HelpText<"Allow variadic functions in CUDA device code.">,
  MarshallingInfoFlag<LangOpts<"CUDAAllowVariadicFunctions">>;
def fno_cuda_host_device_constexpr : Flag<["-"], "fno-cuda-host-device-constexpr">,
  HelpText<"Don't treat unattributed constexpr functions as __host__ __device__.">,
  MarshallingInfoNegativeFlag<LangOpts<"CUDAHostDeviceConstexpr">>;

} // let Flags = [CC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// OpenMP Options
//===----------------------------------------------------------------------===//

def fopenmp_is_device : Flag<["-"], "fopenmp-is-device">,
  HelpText<"Generate code only for an OpenMP target device.">,
  Flags<[CC1Option, NoDriverOption]>;
def fopenmp_host_ir_file_path : Separate<["-"], "fopenmp-host-ir-file-path">,
  HelpText<"Path to the IR file produced by the frontend for the host.">,
  Flags<[CC1Option, NoDriverOption]>;

//===----------------------------------------------------------------------===//
// SYCL Options
//===----------------------------------------------------------------------===//

def fsycl_is_device : Flag<["-"], "fsycl-is-device">,
  HelpText<"Generate code for SYCL device.">,
  Flags<[CC1Option, NoDriverOption]>,
  MarshallingInfoFlag<LangOpts<"SYCLIsDevice">>;
def fsycl_is_host : Flag<["-"], "fsycl-is-host">,
  HelpText<"SYCL host compilation">,
  Flags<[CC1Option, NoDriverOption]>,
  MarshallingInfoFlag<LangOpts<"SYCLIsHost">>;

def sycl_std_EQ : Joined<["-"], "sycl-std=">, Group<sycl_Group>,
  Flags<[CC1Option, NoArgumentUnused, CoreOption]>,
  HelpText<"SYCL language standard to compile for.">,
  Values<"2020,2017,121,1.2.1,sycl-1.2.1">,
  NormalizedValues<["SYCL_2020", "SYCL_2017", "SYCL_2017", "SYCL_2017", "SYCL_2017"]>,
  NormalizedValuesScope<"LangOptions">,
  MarshallingInfoEnum<LangOpts<"SYCLVersion">, "SYCL_None">,
  ShouldParseIf<!strconcat(fsycl_is_device.KeyPath, "||", fsycl_is_host.KeyPath)>;

defm cuda_approx_transcendentals : BoolFOption<"cuda-approx-transcendentals",
  LangOpts<"CUDADeviceApproxTranscendentals">, DefaultFalse,
  PosFlag<SetTrue, [CC1Option], "Use">, NegFlag<SetFalse, [], "Don't use">,
  BothFlags<[], " approximate transcendental functions">>,
  ShouldParseIf<fcuda_is_device.KeyPath>;

//===----------------------------------------------------------------------===//
// Frontend Options - cc1 + fc1
//===----------------------------------------------------------------------===//

let Flags = [CC1Option, FC1Option, NoDriverOption] in {
let Group = Action_Group in {

def emit_obj : Flag<["-"], "emit-obj">,
  HelpText<"Emit native object files">;
def init_only : Flag<["-"], "init-only">,
  HelpText<"Only execute frontend initialization">;
def emit_llvm_bc : Flag<["-"], "emit-llvm-bc">,
  HelpText<"Build ASTs then convert to LLVM, emit .bc file">;

} // let Group = Action_Group

def load : Separate<["-"], "load">, MetaVarName<"<dsopath>">,
  HelpText<"Load the named plugin (dynamic shared object)">;
def plugin : Separate<["-"], "plugin">, MetaVarName<"<name>">,
  HelpText<"Use the named plugin action instead of the default action (use \"help\" to list available options)">;
defm debug_pass_manager : BoolOption<"f", "debug-pass-manager",
  CodeGenOpts<"DebugPassManager">, DefaultFalse,
  PosFlag<SetTrue, [], "Prints debug information for the new pass manager">,
  NegFlag<SetFalse, [], "Disables debug printing for the new pass manager">>;

} // let Flags = [CC1Option, FC1Option, NoDriverOption]

//===----------------------------------------------------------------------===//
// cc1as-only Options
//===----------------------------------------------------------------------===//

let Flags = [CC1AsOption, NoDriverOption] in {

// Language Options
def n : Flag<["-"], "n">,
  HelpText<"Don't automatically start assembly file with a text section">;

// Frontend Options
def filetype : Separate<["-"], "filetype">,
    HelpText<"Specify the output file type ('asm', 'null', or 'obj')">;

// Transliterate Options
def output_asm_variant : Separate<["-"], "output-asm-variant">,
    HelpText<"Select the asm variant index to use for output">;
def show_encoding : Flag<["-"], "show-encoding">,
    HelpText<"Show instruction encoding information in transliterate mode">;
def show_inst : Flag<["-"], "show-inst">,
    HelpText<"Show internal instruction representation in transliterate mode">;

// Assemble Options
def dwarf_debug_producer : Separate<["-"], "dwarf-debug-producer">,
  HelpText<"The string to embed in the Dwarf debug AT_producer record.">;

def defsym : Separate<["-"], "defsym">,
  HelpText<"Define a value for a symbol">;

} // let Flags = [CC1AsOption]

//===----------------------------------------------------------------------===//
// clang-cl Options
//===----------------------------------------------------------------------===//

def cl_Group : OptionGroup<"<clang-cl options>">, Flags<[CLDXCOption]>,
  HelpText<"CL.EXE COMPATIBILITY OPTIONS">;

def cl_compile_Group : OptionGroup<"<clang-cl compile-only options>">,
  Group<cl_Group>;

def cl_ignored_Group : OptionGroup<"<clang-cl ignored options>">,
  Group<cl_Group>;

class CLFlag<string name> : Option<["/", "-"], name, KIND_FLAG>,
  Group<cl_Group>, Flags<[CLOption, NoXarchOption]>;

class CLCompileFlag<string name> : Option<["/", "-"], name, KIND_FLAG>,
  Group<cl_compile_Group>, Flags<[CLOption, NoXarchOption]>;

class CLIgnoredFlag<string name> : Option<["/", "-"], name, KIND_FLAG>,
  Group<cl_ignored_Group>, Flags<[CLOption, NoXarchOption]>;

class CLJoined<string name> : Option<["/", "-"], name, KIND_JOINED>,
  Group<cl_Group>, Flags<[CLOption, NoXarchOption]>;

class CLCompileJoined<string name> : Option<["/", "-"], name, KIND_JOINED>,
  Group<cl_compile_Group>, Flags<[CLOption, NoXarchOption]>;

class CLIgnoredJoined<string name> : Option<["/", "-"], name, KIND_JOINED>,
  Group<cl_ignored_Group>, Flags<[CLOption, NoXarchOption, HelpHidden]>;

class CLJoinedOrSeparate<string name> : Option<["/", "-"], name,
  KIND_JOINED_OR_SEPARATE>, Group<cl_Group>, Flags<[CLOption, NoXarchOption]>;

class CLDXCJoinedOrSeparate<string name> : Option<["/", "-"], name,
  KIND_JOINED_OR_SEPARATE>, Group<cl_Group>, Flags<[CLDXCOption, NoXarchOption]>;

class CLCompileJoinedOrSeparate<string name> : Option<["/", "-"], name,
  KIND_JOINED_OR_SEPARATE>, Group<cl_compile_Group>,
  Flags<[CLOption, NoXarchOption]>;

class CLRemainingArgsJoined<string name> : Option<["/", "-"], name,
  KIND_REMAINING_ARGS_JOINED>, Group<cl_Group>, Flags<[CLOption, NoXarchOption]>;

// Aliases:
// (We don't put any of these in cl_compile_Group as the options they alias are
// already in the right group.)

def _SLASH_Brepro : CLFlag<"Brepro">,
  HelpText<"Do not write current time into COFF output (breaks link.exe /incremental)">,
  Alias<mno_incremental_linker_compatible>;
def _SLASH_Brepro_ : CLFlag<"Brepro-">,
  HelpText<"Write current time into COFF output (default)">,
  Alias<mincremental_linker_compatible>;
def _SLASH_C : CLFlag<"C">,
  HelpText<"Do not discard comments when preprocessing">, Alias<C>;
def _SLASH_c : CLFlag<"c">, HelpText<"Compile only">, Alias<c>;
def _SLASH_d1PP : CLFlag<"d1PP">,
  HelpText<"Retain macro definitions in /E mode">, Alias<dD>;
def _SLASH_d1reportAllClassLayout : CLFlag<"d1reportAllClassLayout">,
  HelpText<"Dump record layout information">,
  Alias<Xclang>, AliasArgs<["-fdump-record-layouts"]>;
def _SLASH_diagnostics_caret : CLFlag<"diagnostics:caret">,
  HelpText<"Enable caret and column diagnostics (default)">;
def _SLASH_diagnostics_column : CLFlag<"diagnostics:column">,
  HelpText<"Disable caret diagnostics but keep column info">;
def _SLASH_diagnostics_classic : CLFlag<"diagnostics:classic">,
  HelpText<"Disable column and caret diagnostics">;
def _SLASH_D : CLJoinedOrSeparate<"D">, HelpText<"Define macro">,
  MetaVarName<"<macro[=value]>">, Alias<D>;
def _SLASH_E : CLFlag<"E">, HelpText<"Preprocess to stdout">, Alias<E>;
def _SLASH_external_COLON_I : CLJoinedOrSeparate<"external:I">, Alias<isystem>,
  HelpText<"Add directory to include search path with warnings suppressed">,
  MetaVarName<"<dir>">;
def _SLASH_fp_except : CLFlag<"fp:except">, HelpText<"">, Alias<ftrapping_math>;
def _SLASH_fp_except_ : CLFlag<"fp:except-">,
  HelpText<"">, Alias<fno_trapping_math>;
def _SLASH_fp_fast : CLFlag<"fp:fast">, HelpText<"">, Alias<ffast_math>;
def _SLASH_fp_precise : CLFlag<"fp:precise">,
  HelpText<"">, Alias<fno_fast_math>;
def _SLASH_fp_strict : CLFlag<"fp:strict">, HelpText<"">, Alias<fno_fast_math>;
def _SLASH_fsanitize_EQ_address : CLFlag<"fsanitize=address">,
  HelpText<"Enable AddressSanitizer">,
  Alias<fsanitize_EQ>, AliasArgs<["address"]>;
def _SLASH_GA : CLFlag<"GA">, Alias<ftlsmodel_EQ>, AliasArgs<["local-exec"]>,
  HelpText<"Assume thread-local variables are defined in the executable">;
def _SLASH_GR : CLFlag<"GR">, HelpText<"Emit RTTI data (default)">;
def _SLASH_GR_ : CLFlag<"GR-">, HelpText<"Do not emit RTTI data">;
def _SLASH_GF : CLIgnoredFlag<"GF">,
  HelpText<"Enable string pooling (default)">;
def _SLASH_GF_ : CLFlag<"GF-">, HelpText<"Disable string pooling">,
  Alias<fwritable_strings>;
def _SLASH_GS : CLFlag<"GS">,
  HelpText<"Enable buffer security check (default)">;
def _SLASH_GS_ : CLFlag<"GS-">, HelpText<"Disable buffer security check">;
def : CLFlag<"Gs">, HelpText<"Use stack probes (default)">,
  Alias<mstack_probe_size>, AliasArgs<["4096"]>;
def _SLASH_Gs : CLJoined<"Gs">,
  HelpText<"Set stack probe size (default 4096)">, Alias<mstack_probe_size>;
def _SLASH_Gy : CLFlag<"Gy">, HelpText<"Put each function in its own section">,
  Alias<ffunction_sections>;
def _SLASH_Gy_ : CLFlag<"Gy-">,
  HelpText<"Do not put each function in its own section (default)">,
  Alias<fno_function_sections>;
def _SLASH_Gw : CLFlag<"Gw">, HelpText<"Put each data item in its own section">,
  Alias<fdata_sections>;
def _SLASH_Gw_ : CLFlag<"Gw-">,
  HelpText<"Do not put each data item in its own section (default)">,
  Alias<fno_data_sections>;
def _SLASH_help : CLFlag<"help">, Alias<help>,
  HelpText<"Display available options">;
def _SLASH_HELP : CLFlag<"HELP">, Alias<help>;
def _SLASH_hotpatch : CLFlag<"hotpatch">, Alias<fms_hotpatch>,
  HelpText<"Create hotpatchable image">;
def _SLASH_I : CLDXCJoinedOrSeparate<"I">,
  HelpText<"Add directory to include search path">, MetaVarName<"<dir>">,
  Alias<I>;
def _SLASH_J : CLFlag<"J">, HelpText<"Make char type unsigned">,
  Alias<funsigned_char>;

// The _SLASH_O option handles all the /O flags, but we also provide separate
// aliased options to provide separate help messages.
def _SLASH_O : CLJoined<"O">,
  HelpText<"Set multiple /O flags at once; e.g. '/O2y-' for '/O2 /Oy-'">,
  MetaVarName<"<flags>">;
def : CLFlag<"O1">, Alias<_SLASH_O>, AliasArgs<["1"]>,
  HelpText<"Optimize for size  (like /Og     /Os /Oy /Ob2 /GF /Gy)">;
def : CLFlag<"O2">, Alias<_SLASH_O>, AliasArgs<["2"]>,
  HelpText<"Optimize for speed (like /Og /Oi /Ot /Oy /Ob2 /GF /Gy)">;
def : CLFlag<"Ob0">, Alias<_SLASH_O>, AliasArgs<["b0"]>,
  HelpText<"Disable function inlining">;
def : CLFlag<"Ob1">, Alias<_SLASH_O>, AliasArgs<["b1"]>,
  HelpText<"Only inline functions explicitly or implicitly marked inline">;
def : CLFlag<"Ob2">, Alias<_SLASH_O>, AliasArgs<["b2"]>,
  HelpText<"Inline functions as deemed beneficial by the compiler">;
def : CLFlag<"Od">, Alias<_SLASH_O>, AliasArgs<["d"]>,
  HelpText<"Disable optimization">;
def : CLFlag<"Og">, Alias<_SLASH_O>, AliasArgs<["g"]>,
  HelpText<"No effect">;
def : CLFlag<"Oi">, Alias<_SLASH_O>, AliasArgs<["i"]>,
  HelpText<"Enable use of builtin functions">;
def : CLFlag<"Oi-">, Alias<_SLASH_O>, AliasArgs<["i-"]>,
  HelpText<"Disable use of builtin functions">;
def : CLFlag<"Os">, Alias<_SLASH_O>, AliasArgs<["s"]>,
  HelpText<"Optimize for size">;
def : CLFlag<"Ot">, Alias<_SLASH_O>, AliasArgs<["t"]>,
  HelpText<"Optimize for speed">;
def : CLFlag<"Ox">, Alias<_SLASH_O>, AliasArgs<["x"]>,
  HelpText<"Deprecated (like /Og /Oi /Ot /Oy /Ob2); use /O2">;
def : CLFlag<"Oy">, Alias<_SLASH_O>, AliasArgs<["y"]>,
  HelpText<"Enable frame pointer omission (x86 only)">;
def : CLFlag<"Oy-">, Alias<_SLASH_O>, AliasArgs<["y-"]>,
  HelpText<"Disable frame pointer omission (x86 only, default)">;

def _SLASH_QUESTION : CLFlag<"?">, Alias<help>,
  HelpText<"Display available options">;
def _SLASH_Qvec : CLFlag<"Qvec">,
  HelpText<"Enable the loop vectorization passes">, Alias<fvectorize>;
def _SLASH_Qvec_ : CLFlag<"Qvec-">,
  HelpText<"Disable the loop vectorization passes">, Alias<fno_vectorize>;
def _SLASH_showIncludes : CLFlag<"showIncludes">,
  HelpText<"Print info about included files to stderr">;
def _SLASH_showIncludes_user : CLFlag<"showIncludes:user">,
  HelpText<"Like /showIncludes but omit system headers">;
def _SLASH_showFilenames : CLFlag<"showFilenames">,
  HelpText<"Print the name of each compiled file">;
def _SLASH_showFilenames_ : CLFlag<"showFilenames-">,
  HelpText<"Do not print the name of each compiled file (default)">;
def _SLASH_source_charset : CLCompileJoined<"source-charset:">,
  HelpText<"Set source encoding, supports only UTF-8">,
  Alias<finput_charset_EQ>;
def _SLASH_execution_charset : CLCompileJoined<"execution-charset:">,
  HelpText<"Set runtime encoding, supports only UTF-8">,
  Alias<fexec_charset_EQ>;
def _SLASH_std : CLCompileJoined<"std:">,
  HelpText<"Set language version (c++14,c++17,c++20,c++latest,c11,c17)">;
def _SLASH_U : CLJoinedOrSeparate<"U">, HelpText<"Undefine macro">,
  MetaVarName<"<macro>">, Alias<U>;
def _SLASH_validate_charset : CLFlag<"validate-charset">,
  Alias<W_Joined>, AliasArgs<["invalid-source-encoding"]>;
def _SLASH_validate_charset_ : CLFlag<"validate-charset-">,
  Alias<W_Joined>, AliasArgs<["no-invalid-source-encoding"]>;
def _SLASH_external_W0 : CLFlag<"external:W0">, HelpText<"Ignore warnings from system headers (default)">, Alias<Wno_system_headers>;
def _SLASH_external_W1 : CLFlag<"external:W1">, HelpText<"Enable -Wsystem-headers">, Alias<Wsystem_headers>;
def _SLASH_external_W2 : CLFlag<"external:W2">, HelpText<"Enable -Wsystem-headers">, Alias<Wsystem_headers>;
def _SLASH_external_W3 : CLFlag<"external:W3">, HelpText<"Enable -Wsystem-headers">, Alias<Wsystem_headers>;
def _SLASH_external_W4 : CLFlag<"external:W4">, HelpText<"Enable -Wsystem-headers">, Alias<Wsystem_headers>;
def _SLASH_W0 : CLFlag<"W0">, HelpText<"Disable all warnings">, Alias<w>;
def _SLASH_W1 : CLFlag<"W1">, HelpText<"Enable -Wall">, Alias<Wall>;
def _SLASH_W2 : CLFlag<"W2">, HelpText<"Enable -Wall">, Alias<Wall>;
def _SLASH_W3 : CLFlag<"W3">, HelpText<"Enable -Wall">, Alias<Wall>;
def _SLASH_W4 : CLFlag<"W4">, HelpText<"Enable -Wall and -Wextra">, Alias<WCL4>;
def _SLASH_Wall : CLFlag<"Wall">, HelpText<"Enable -Weverything">,
  Alias<W_Joined>, AliasArgs<["everything"]>;
def _SLASH_WX : CLFlag<"WX">, HelpText<"Treat warnings as errors">,
  Alias<W_Joined>, AliasArgs<["error"]>;
def _SLASH_WX_ : CLFlag<"WX-">,
  HelpText<"Do not treat warnings as errors (default)">,
  Alias<W_Joined>, AliasArgs<["no-error"]>;
def _SLASH_w_flag : CLFlag<"w">, HelpText<"Disable all warnings">, Alias<w>;
def _SLASH_wd : CLCompileJoined<"wd">;
def _SLASH_vd : CLJoined<"vd">, HelpText<"Control vtordisp placement">,
  Alias<vtordisp_mode_EQ>;
def _SLASH_X : CLFlag<"X">,
  HelpText<"Do not add %INCLUDE% to include search path">, Alias<nostdlibinc>;
def _SLASH_Zc_sizedDealloc : CLFlag<"Zc:sizedDealloc">,
  HelpText<"Enable C++14 sized global deallocation functions">,
  Alias<fsized_deallocation>;
def _SLASH_Zc_sizedDealloc_ : CLFlag<"Zc:sizedDealloc-">,
  HelpText<"Disable C++14 sized global deallocation functions">,
  Alias<fno_sized_deallocation>;
def _SLASH_Zc_alignedNew : CLFlag<"Zc:alignedNew">,
  HelpText<"Enable C++17 aligned allocation functions">,
  Alias<faligned_allocation>;
def _SLASH_Zc_alignedNew_ : CLFlag<"Zc:alignedNew-">,
  HelpText<"Disable C++17 aligned allocation functions">,
  Alias<fno_aligned_allocation>;
def _SLASH_Zc_char8_t : CLFlag<"Zc:char8_t">,
  HelpText<"Enable char8_t from C++2a">,
  Alias<fchar8__t>;
def _SLASH_Zc_char8_t_ : CLFlag<"Zc:char8_t-">,
  HelpText<"Disable char8_t from c++2a">,
  Alias<fno_char8__t>;
def _SLASH_Zc_strictStrings : CLFlag<"Zc:strictStrings">,
  HelpText<"Treat string literals as const">, Alias<W_Joined>,
  AliasArgs<["error=c++11-compat-deprecated-writable-strings"]>;
def _SLASH_Zc_threadSafeInit : CLFlag<"Zc:threadSafeInit">,
  HelpText<"Enable thread-safe initialization of static variables">,
  Alias<fthreadsafe_statics>;
def _SLASH_Zc_threadSafeInit_ : CLFlag<"Zc:threadSafeInit-">,
  HelpText<"Disable thread-safe initialization of static variables">,
  Alias<fno_threadsafe_statics>;
def _SLASH_Zc_trigraphs : CLFlag<"Zc:trigraphs">,
  HelpText<"Enable trigraphs">, Alias<ftrigraphs>;
def _SLASH_Zc_trigraphs_off : CLFlag<"Zc:trigraphs-">,
  HelpText<"Disable trigraphs (default)">, Alias<fno_trigraphs>;
def _SLASH_Zc_twoPhase : CLFlag<"Zc:twoPhase">,
  HelpText<"Enable two-phase name lookup in templates">,
  Alias<fno_delayed_template_parsing>;
def _SLASH_Zc_twoPhase_ : CLFlag<"Zc:twoPhase-">,
  HelpText<"Disable two-phase name lookup in templates (default)">,
  Alias<fdelayed_template_parsing>;
def _SLASH_Zc_wchar_t : CLFlag<"Zc:wchar_t">,
  HelpText<"Enable C++ builtin type wchar_t (default)">;
def _SLASH_Zc_wchar_t_ : CLFlag<"Zc:wchar_t-">,
  HelpText<"Disable C++ builtin type wchar_t">;
def _SLASH_Z7 : CLFlag<"Z7">,
  HelpText<"Enable CodeView debug information in object files">;
def _SLASH_Zi : CLFlag<"Zi">, Alias<_SLASH_Z7>,
  HelpText<"Like /Z7">;
def _SLASH_Zp : CLJoined<"Zp">,
  HelpText<"Set default maximum struct packing alignment">,
  Alias<fpack_struct_EQ>;
def _SLASH_Zp_flag : CLFlag<"Zp">,
  HelpText<"Set default maximum struct packing alignment to 1">,
  Alias<fpack_struct_EQ>, AliasArgs<["1"]>;
def _SLASH_Zs : CLFlag<"Zs">, HelpText<"Syntax-check only">,
  Alias<fsyntax_only>;
def _SLASH_openmp_ : CLFlag<"openmp-">,
  HelpText<"Disable OpenMP support">, Alias<fno_openmp>;
def _SLASH_openmp : CLFlag<"openmp">, HelpText<"Enable OpenMP support">,
  Alias<fopenmp>;
def _SLASH_openmp_experimental : CLFlag<"openmp:experimental">,
  HelpText<"Enable OpenMP support with experimental SIMD support">,
  Alias<fopenmp>;
def _SLASH_tune : CLCompileJoined<"tune:">,
  HelpText<"Set CPU for optimization without affecting instruction set">,
  Alias<mtune_EQ>;
def _SLASH_QIntel_jcc_erratum : CLFlag<"QIntel-jcc-erratum">,
  HelpText<"Align branches within 32-byte boundaries to mitigate the performance impact of the Intel JCC erratum.">,
  Alias<mbranches_within_32B_boundaries>;

// Non-aliases:

def _SLASH_arch : CLCompileJoined<"arch:">,
  HelpText<"Set architecture for code generation">;

def _SLASH_M_Group : OptionGroup<"</M group>">, Group<cl_compile_Group>;
def _SLASH_volatile_Group : OptionGroup<"</volatile group>">,
  Group<cl_compile_Group>;

def _SLASH_EH : CLJoined<"EH">, HelpText<"Set exception handling model">;
def _SLASH_EP : CLFlag<"EP">,
  HelpText<"Disable linemarker output and preprocess to stdout">;
def _SLASH_external_env : CLJoined<"external:env:">,
  HelpText<"Add dirs in env var <var> to include search path with warnings suppressed">,
  MetaVarName<"<var>">;
def _SLASH_FA : CLJoined<"FA">,
  HelpText<"Output assembly code file during compilation">;
def _SLASH_Fa : CLJoined<"Fa">,
  HelpText<"Set assembly output file name (with /FA)">,
  MetaVarName<"<file or dir/>">;
def _SLASH_FI : CLJoinedOrSeparate<"FI">,
  HelpText<"Include file before parsing">, Alias<include_>;
def _SLASH_Fe : CLJoined<"Fe">,
  HelpText<"Set output executable file name">,
  MetaVarName<"<file or dir/>">;
def _SLASH_Fe_COLON : CLJoined<"Fe:">, Alias<_SLASH_Fe>;
def _SLASH_Fi : CLCompileJoined<"Fi">,
  HelpText<"Set preprocess output file name (with /P)">,
  MetaVarName<"<file>">;
def _SLASH_Fo : CLCompileJoined<"Fo">,
  HelpText<"Set output object file (with /c)">,
  MetaVarName<"<file or dir/>">;
def _SLASH_guard : CLJoined<"guard:">,
  HelpText<"Enable Control Flow Guard with /guard:cf, or only the table with /guard:cf,nochecks. "
           "Enable EH Continuation Guard with /guard:ehcont">;
def _SLASH_GX : CLFlag<"GX">,
  HelpText<"Deprecated; use /EHsc">;
def _SLASH_GX_ : CLFlag<"GX-">,
  HelpText<"Deprecated (like not passing /EH)">;
def _SLASH_imsvc : CLJoinedOrSeparate<"imsvc">,
  HelpText<"Add <dir> to system include search path, as if in %INCLUDE%">,
  MetaVarName<"<dir>">;
def _SLASH_JMC : CLFlag<"JMC">,
  HelpText<"Enable just-my-code debugging">;
def _SLASH_JMC_ : CLFlag<"JMC-">,
  HelpText<"Disable just-my-code debugging (default)">;
def _SLASH_LD : CLFlag<"LD">, HelpText<"Create DLL">;
def _SLASH_LDd : CLFlag<"LDd">, HelpText<"Create debug DLL">;
def _SLASH_link : CLRemainingArgsJoined<"link">,
  HelpText<"Forward options to the linker">, MetaVarName<"<options>">;
def _SLASH_MD : Option<["/", "-"], "MD", KIND_FLAG>, Group<_SLASH_M_Group>,
  Flags<[CLOption, NoXarchOption]>, HelpText<"Use DLL run-time">;
def _SLASH_MDd : Option<["/", "-"], "MDd", KIND_FLAG>, Group<_SLASH_M_Group>,
  Flags<[CLOption, NoXarchOption]>, HelpText<"Use DLL debug run-time">;
def _SLASH_MT : Option<["/", "-"], "MT", KIND_FLAG>, Group<_SLASH_M_Group>,
  Flags<[CLOption, NoXarchOption]>, HelpText<"Use static run-time">;
def _SLASH_MTd : Option<["/", "-"], "MTd", KIND_FLAG>, Group<_SLASH_M_Group>,
  Flags<[CLOption, NoXarchOption]>, HelpText<"Use static debug run-time">;
def _SLASH_o : CLJoinedOrSeparate<"o">,
  HelpText<"Deprecated (set output file name); use /Fe or /Fe">,
  MetaVarName<"<file or dir/>">;
def _SLASH_P : CLFlag<"P">, HelpText<"Preprocess to file">;
def _SLASH_permissive : CLFlag<"permissive">,
  HelpText<"Enable some non conforming code to compile">;
def _SLASH_permissive_ : CLFlag<"permissive-">,
  HelpText<"Disable non conforming code from compiling (default)">;
def _SLASH_Tc : CLCompileJoinedOrSeparate<"Tc">,
  HelpText<"Treat <file> as C source file">, MetaVarName<"<file>">;
def _SLASH_TC : CLCompileFlag<"TC">, HelpText<"Treat all source files as C">;
def _SLASH_Tp : CLCompileJoinedOrSeparate<"Tp">,
  HelpText<"Treat <file> as C++ source file">, MetaVarName<"<file>">;
def _SLASH_TP : CLCompileFlag<"TP">, HelpText<"Treat all source files as C++">;
def _SLASH_diasdkdir : CLJoinedOrSeparate<"diasdkdir">,
  HelpText<"Path to the DIA SDK">, MetaVarName<"<dir>">;
def _SLASH_vctoolsdir : CLJoinedOrSeparate<"vctoolsdir">,
  HelpText<"Path to the VCToolChain">, MetaVarName<"<dir>">;
def _SLASH_vctoolsversion : CLJoinedOrSeparate<"vctoolsversion">,
  HelpText<"For use with /winsysroot, defaults to newest found">;
def _SLASH_winsdkdir : CLJoinedOrSeparate<"winsdkdir">,
  HelpText<"Path to the Windows SDK">, MetaVarName<"<dir>">;
def _SLASH_winsdkversion : CLJoinedOrSeparate<"winsdkversion">,
  HelpText<"Full version of the Windows SDK, defaults to newest found">;
def _SLASH_winsysroot : CLJoinedOrSeparate<"winsysroot">,
  HelpText<"Same as \"/diasdkdir <dir>/DIA SDK\" /vctoolsdir <dir>/VC/Tools/MSVC/<vctoolsversion> \"/winsdkdir <dir>/Windows Kits/10\"">,
  MetaVarName<"<dir>">;
def _SLASH_volatile_iso : Option<["/", "-"], "volatile:iso", KIND_FLAG>,
  Group<_SLASH_volatile_Group>, Flags<[CLOption, NoXarchOption]>,
  HelpText<"Volatile loads and stores have standard semantics">;
def _SLASH_vmb : CLFlag<"vmb">,
  HelpText<"Use a best-case representation method for member pointers">;
def _SLASH_vmg : CLFlag<"vmg">,
  HelpText<"Use a most-general representation for member pointers">;
def _SLASH_vms : CLFlag<"vms">,
  HelpText<"Set the default most-general representation to single inheritance">;
def _SLASH_vmm : CLFlag<"vmm">,
  HelpText<"Set the default most-general representation to "
           "multiple inheritance">;
def _SLASH_vmv : CLFlag<"vmv">,
  HelpText<"Set the default most-general representation to "
           "virtual inheritance">;
def _SLASH_volatile_ms  : Option<["/", "-"], "volatile:ms", KIND_FLAG>,
  Group<_SLASH_volatile_Group>, Flags<[CLOption, NoXarchOption]>,
  HelpText<"Volatile loads and stores have acquire and release semantics">;
def _SLASH_clang : CLJoined<"clang:">,
  HelpText<"Pass <arg> to the clang driver">, MetaVarName<"<arg>">;
def _SLASH_Zl : CLFlag<"Zl">,
  HelpText<"Do not let object file auto-link default libraries">;

def _SLASH_Yc : CLJoined<"Yc">,
  HelpText<"Generate a pch file for all code up to and including <filename>">,
  MetaVarName<"<filename>">;
def _SLASH_Yu : CLJoined<"Yu">,
  HelpText<"Load a pch file and use it instead of all code up to "
           "and including <filename>">,
  MetaVarName<"<filename>">;
def _SLASH_Y_ : CLFlag<"Y-">,
  HelpText<"Disable precompiled headers, overrides /Yc and /Yu">;
def _SLASH_Zc_dllexportInlines : CLFlag<"Zc:dllexportInlines">,
  HelpText<"dllexport/dllimport inline member functions of dllexport/import classes (default)">;
def _SLASH_Zc_dllexportInlines_ : CLFlag<"Zc:dllexportInlines-">,
  HelpText<"Do not dllexport/dllimport inline member functions of dllexport/import classes">;
def _SLASH_Fp : CLJoined<"Fp">,
  HelpText<"Set pch file name (with /Yc and /Yu)">, MetaVarName<"<file>">;

def _SLASH_Gd : CLFlag<"Gd">,
  HelpText<"Set __cdecl as a default calling convention">;
def _SLASH_Gr : CLFlag<"Gr">,
  HelpText<"Set __fastcall as a default calling convention">;
def _SLASH_Gz : CLFlag<"Gz">,
  HelpText<"Set __stdcall as a default calling convention">;
def _SLASH_Gv : CLFlag<"Gv">,
  HelpText<"Set __vectorcall as a default calling convention">;
def _SLASH_Gregcall : CLFlag<"Gregcall">,
  HelpText<"Set __regcall as a default calling convention">;

// Ignored:

def _SLASH_analyze_ : CLIgnoredFlag<"analyze-">;
def _SLASH_bigobj : CLIgnoredFlag<"bigobj">;
def _SLASH_cgthreads : CLIgnoredJoined<"cgthreads">;
def _SLASH_d2FastFail : CLIgnoredFlag<"d2FastFail">;
def _SLASH_d2Zi_PLUS : CLIgnoredFlag<"d2Zi+">;
def _SLASH_errorReport : CLIgnoredJoined<"errorReport">;
def _SLASH_FC : CLIgnoredFlag<"FC">;
def _SLASH_Fd : CLIgnoredJoined<"Fd">;
def _SLASH_FS : CLIgnoredFlag<"FS">;
def _SLASH_kernel_ : CLIgnoredFlag<"kernel-">;
def _SLASH_nologo : CLIgnoredFlag<"nologo">;
def _SLASH_RTC : CLIgnoredJoined<"RTC">;
def _SLASH_sdl : CLIgnoredFlag<"sdl">;
def _SLASH_sdl_ : CLIgnoredFlag<"sdl-">;
def _SLASH_utf8 : CLIgnoredFlag<"utf-8">,
  HelpText<"Set source and runtime encoding to UTF-8 (default)">;
def _SLASH_w : CLIgnoredJoined<"w">;
def _SLASH_Wv_ : CLIgnoredJoined<"Wv">;
def _SLASH_Zc___cplusplus : CLIgnoredFlag<"Zc:__cplusplus">;
def _SLASH_Zc_auto : CLIgnoredFlag<"Zc:auto">;
def _SLASH_Zc_forScope : CLIgnoredFlag<"Zc:forScope">;
def _SLASH_Zc_inline : CLIgnoredFlag<"Zc:inline">;
def _SLASH_Zc_rvalueCast : CLIgnoredFlag<"Zc:rvalueCast">;
def _SLASH_Zc_ternary : CLIgnoredFlag<"Zc:ternary">;
def _SLASH_ZH_MD5 : CLIgnoredFlag<"ZH:MD5">;
def _SLASH_ZH_SHA1 : CLIgnoredFlag<"ZH:SHA1">;
def _SLASH_ZH_SHA_256 : CLIgnoredFlag<"ZH:SHA_256">;
def _SLASH_Zm : CLIgnoredJoined<"Zm">;
def _SLASH_Zo : CLIgnoredFlag<"Zo">;
def _SLASH_Zo_ : CLIgnoredFlag<"Zo-">;


// Unsupported:

def _SLASH_await : CLFlag<"await">;
def _SLASH_await_COLON : CLJoined<"await:">;
def _SLASH_constexpr : CLJoined<"constexpr:">;
def _SLASH_AI : CLJoinedOrSeparate<"AI">;
def _SLASH_Bt : CLFlag<"Bt">;
def _SLASH_Bt_plus : CLFlag<"Bt+">;
def _SLASH_clr : CLJoined<"clr">;
def _SLASH_d2 : CLJoined<"d2">;
def _SLASH_doc : CLJoined<"doc">;
def _SLASH_experimental : CLJoined<"experimental:">;
def _SLASH_exportHeader : CLFlag<"exportHeader">;
def _SLASH_external : CLJoined<"external:">;
def _SLASH_favor : CLJoined<"favor">;
def _SLASH_fsanitize_address_use_after_return : CLJoined<"fsanitize-address-use-after-return">;
def _SLASH_fno_sanitize_address_vcasan_lib : CLJoined<"fno-sanitize-address-vcasan-lib">;
def _SLASH_F : CLJoinedOrSeparate<"F">;
def _SLASH_Fm : CLJoined<"Fm">;
def _SLASH_Fr : CLJoined<"Fr">;
def _SLASH_FR : CLJoined<"FR">;
def _SLASH_FU : CLJoinedOrSeparate<"FU">;
def _SLASH_Fx : CLFlag<"Fx">;
def _SLASH_G1 : CLFlag<"G1">;
def _SLASH_G2 : CLFlag<"G2">;
def _SLASH_Ge : CLFlag<"Ge">;
def _SLASH_Gh : CLFlag<"Gh">;
def _SLASH_GH : CLFlag<"GH">;
def _SLASH_GL : CLFlag<"GL">;
def _SLASH_GL_ : CLFlag<"GL-">;
def _SLASH_Gm : CLFlag<"Gm">;
def _SLASH_Gm_ : CLFlag<"Gm-">;
def _SLASH_GT : CLFlag<"GT">;
def _SLASH_GZ : CLFlag<"GZ">;
def _SLASH_H : CLFlag<"H">;
def _SLASH_headername : CLJoined<"headerName:">;
def _SLASH_headerUnit : CLJoinedOrSeparate<"headerUnit">;
def _SLASH_headerUnitAngle : CLJoinedOrSeparate<"headerUnit:angle">;
def _SLASH_headerUnitQuote : CLJoinedOrSeparate<"headerUnit:quote">;
def _SLASH_homeparams : CLFlag<"homeparams">;
def _SLASH_kernel : CLFlag<"kernel">;
def _SLASH_LN : CLFlag<"LN">;
def _SLASH_MP : CLJoined<"MP">;
def _SLASH_Qfast_transcendentals : CLFlag<"Qfast_transcendentals">;
def _SLASH_QIfist : CLFlag<"QIfist">;
def _SLASH_Qimprecise_fwaits : CLFlag<"Qimprecise_fwaits">;
def _SLASH_Qpar : CLFlag<"Qpar">;
def _SLASH_Qpar_report : CLJoined<"Qpar-report">;
def _SLASH_Qsafe_fp_loads : CLFlag<"Qsafe_fp_loads">;
def _SLASH_Qspectre : CLFlag<"Qspectre">;
def _SLASH_Qspectre_load : CLFlag<"Qspectre-load">;
def _SLASH_Qspectre_load_cf : CLFlag<"Qspectre-load-cf">;
def _SLASH_Qvec_report : CLJoined<"Qvec-report">;
def _SLASH_reference : CLJoinedOrSeparate<"reference">;
def _SLASH_sourceDependencies : CLJoinedOrSeparate<"sourceDependencies">;
def _SLASH_sourceDependenciesDirectives : CLJoinedOrSeparate<"sourceDependencies:directives">;
def _SLASH_translateInclude : CLFlag<"translateInclude">;
def _SLASH_u : CLFlag<"u">;
def _SLASH_V : CLFlag<"V">;
def _SLASH_WL : CLFlag<"WL">;
def _SLASH_Wp64 : CLFlag<"Wp64">;
def _SLASH_Yd : CLFlag<"Yd">;
def _SLASH_Yl : CLJoined<"Yl">;
def _SLASH_Za : CLFlag<"Za">;
def _SLASH_Zc : CLJoined<"Zc:">;
def _SLASH_Ze : CLFlag<"Ze">;
def _SLASH_Zg : CLFlag<"Zg">;
def _SLASH_ZI : CLFlag<"ZI">;
def _SLASH_ZW : CLJoined<"ZW">;

//===----------------------------------------------------------------------===//
// clang-dxc Options
//===----------------------------------------------------------------------===//

def dxc_Group : OptionGroup<"<clang-dxc options>">, Flags<[DXCOption]>,
  HelpText<"dxc compatibility options">;
class DXCFlag<string name> : Option<["/", "-"], name, KIND_FLAG>,
  Group<dxc_Group>, Flags<[DXCOption, NoXarchOption]>;
class DXCJoinedOrSeparate<string name> : Option<["/", "-"], name,
  KIND_JOINED_OR_SEPARATE>, Group<dxc_Group>, Flags<[DXCOption, NoXarchOption]>;

def dxc_help : Option<["/", "-", "--"], "help", KIND_JOINED>,
  Group<dxc_Group>, Flags<[DXCOption, NoXarchOption]>, Alias<help>,
  HelpText<"Display available options">;
def dxc_no_stdinc : DXCFlag<"hlsl-no-stdinc">,
  HelpText<"HLSL only. Disables all standard includes containing non-native compiler types and functions.">;
def Fo : DXCJoinedOrSeparate<"Fo">, Alias<o>,
  HelpText<"Output object file">;
def dxil_validator_version : Option<["/", "-"], "validator-version", KIND_SEPARATE>,
  Group<dxc_Group>, Flags<[DXCOption, NoXarchOption, CC1Option, HelpHidden]>,
  HelpText<"Override validator version for module. Format: <major.minor>;"
           "Default: DXIL.dll version or current internal version">,
  MarshallingInfoString<TargetOpts<"DxilValidatorVersion">>;
def target_profile : DXCJoinedOrSeparate<"T">, MetaVarName<"<profile>">,
  HelpText<"Set target profile">,
  Values<"ps_6_0, ps_6_1, ps_6_2, ps_6_3, ps_6_4, ps_6_5, ps_6_6, ps_6_7,"
         "vs_6_0, vs_6_1, vs_6_2, vs_6_3, vs_6_4, vs_6_5, vs_6_6, vs_6_7,"
         "gs_6_0, gs_6_1, gs_6_2, gs_6_3, gs_6_4, gs_6_5, gs_6_6, gs_6_7,"
         "hs_6_0, hs_6_1, hs_6_2, hs_6_3, hs_6_4, hs_6_5, hs_6_6, hs_6_7,"
         "ds_6_0, ds_6_1, ds_6_2, ds_6_3, ds_6_4, ds_6_5, ds_6_6, ds_6_7,"
         "cs_6_0, cs_6_1, cs_6_2, cs_6_3, cs_6_4, cs_6_5, cs_6_6, cs_6_7,"
         "lib_6_3, lib_6_4, lib_6_5, lib_6_6, lib_6_7, lib_6_x,"
         "ms_6_5, ms_6_6, ms_6_7,"
         "as_6_5, as_6_6, as_6_7">;
def dxc_D : Option<["--", "/", "-"], "D", KIND_JOINED_OR_SEPARATE>,
  Group<dxc_Group>, Flags<[DXCOption, NoXarchOption]>, Alias<D>;
def emit_pristine_llvm : DXCFlag<"emit-pristine-llvm">,
  HelpText<"Emit pristine LLVM IR from the frontend by not running any LLVM passes at all."
           "Same as -S + -emit-llvm + -disable-llvm-passes.">;
def fcgl : DXCFlag<"fcgl">, Alias<emit_pristine_llvm>;
def enable_16bit_types : DXCFlag<"enable-16bit-types">, Alias<fnative_half_type>,
  HelpText<"Enable 16-bit types and disable min precision types."
           "Available in HLSL 2018 and shader model 6.2.">;
