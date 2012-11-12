//===--- Options.td - Options for clang -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by clang.
//
//===----------------------------------------------------------------------===//

// Include the common option parsing interfaces.
include "OptParser.td"

/////////
// Groups

// Meta-group which defines
def CompileOnly_Group     : OptionGroup<"<CompileOnly group>">;
def Action_Group          : OptionGroup<"<action group>">;

def I_Group               : OptionGroup<"<I group>">, Group<CompileOnly_Group>;
def L_Group               : OptionGroup<"<L group>">, Group<CompileOnly_Group>;
def M_Group               : OptionGroup<"<M group>">, Group<CompileOnly_Group>;
def T_Group               : OptionGroup<"<T group>">;
def O_Group               : OptionGroup<"<O group>">, Group<CompileOnly_Group>;
def W_Group               : OptionGroup<"<W group>">, Group<CompileOnly_Group>;
def X_Group               : OptionGroup<"<X group>">;
def a_Group               : OptionGroup<"<a group>">;
def d_Group               : OptionGroup<"<d group>">;
def f_Group               : OptionGroup<"<f group>">, Group<CompileOnly_Group>;
def f_clang_Group         : OptionGroup<"<f (clang-only) group>">, Group<CompileOnly_Group>;
def g_Group               : OptionGroup<"<g group>">;
def g_flags_Group         : OptionGroup<"<g flags group>">;
def i_Group               : OptionGroup<"<i group>">, Group<CompileOnly_Group>;
def clang_i_Group         : OptionGroup<"<clang i group>">, Group<i_Group>;
def m_Group               : OptionGroup<"<m group>">, Group<CompileOnly_Group>;
def m_x86_Features_Group  : OptionGroup<"<m x86 features group>">, Group<m_Group>;
def m_hexagon_Features_Group  : OptionGroup<"<m hexagon features group>">, Group<m_Group>;
def opencl_Group          : OptionGroup<"<opencl group>">;
def u_Group               : OptionGroup<"<u group>">;
def mips_CPUs_Group       : OptionGroup<"<MIPS CPU aliases group>">,
  Group<CompileOnly_Group>;

def pedantic_Group        : OptionGroup<"<pedantic group>">,
  Group<CompileOnly_Group>;
def reserved_lib_Group   : OptionGroup<"<reserved libs group>">;

// Temporary groups for clang options which we know we don't support,
// but don't want to verbosely warn the user about.
def clang_ignored_f_Group : OptionGroup<"<clang ignored f group>">,
  Group<f_Group>;
def clang_ignored_m_Group : OptionGroup<"<clang ignored m group>">,
  Group<m_Group>;

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
//   # => _HASH
//   , => _COMMA
//   = => _EQ
//   C++ => CXX
//   . => _

// Developer Driver Options

def ccc_Group : OptionGroup<"<clang internal options>">;
def ccc_driver_Group : OptionGroup<"<clang driver internal options>">,
  Group<ccc_Group>, HelpText<"DRIVER OPTIONS">;
def ccc_debug_Group : OptionGroup<"<clang debug/development internal options>">,
  Group<ccc_Group>, HelpText<"DEBUG/DEVELOPMENT OPTIONS">;

class CCCDriverOpt : Group<ccc_driver_Group>, Flags<[DriverOption, HelpHidden]>;
def ccc_cxx : Flag<["-"], "ccc-cxx">, CCCDriverOpt,
  HelpText<"Act as a C++ driver">;
def ccc_echo : Flag<["-"], "ccc-echo">, CCCDriverOpt,
  HelpText<"Echo commands before running them">;
def ccc_gcc_name : Separate<["-"], "ccc-gcc-name">, CCCDriverOpt,
  HelpText<"Name for native GCC compiler">,
  MetaVarName<"<gcc-path>">;
def ccc_clang_archs : Separate<["-"], "ccc-clang-archs">, CCCDriverOpt,
  HelpText<"Comma separate list of architectures to use the clang compiler for">,
  MetaVarName<"<arch-list>">;
def ccc_pch_is_pch : Flag<["-"], "ccc-pch-is-pch">, CCCDriverOpt,
  HelpText<"Use lazy PCH for precompiled headers">;
def ccc_pch_is_pth : Flag<["-"], "ccc-pch-is-pth">, CCCDriverOpt,
  HelpText<"Use pretokenized headers for precompiled headers">;

class CCCDebugOpt : Group<ccc_debug_Group>, Flags<[DriverOption, HelpHidden]>;
def ccc_install_dir : Separate<["-"], "ccc-install-dir">, CCCDebugOpt,
  HelpText<"Simulate installation in the given directory">;
def ccc_print_options : Flag<["-"], "ccc-print-options">, CCCDebugOpt,
  HelpText<"Dump parsed command line arguments">;
def ccc_print_phases : Flag<["-"], "ccc-print-phases">, CCCDebugOpt,
  HelpText<"Dump list of actions to perform">;
def ccc_print_bindings : Flag<["-"], "ccc-print-bindings">, CCCDebugOpt,
  HelpText<"Show bindings of tools to actions">;

def ccc_arcmt_check : Flag<["-"], "ccc-arcmt-check">, CCCDriverOpt,
  HelpText<"Check for ARC migration issues that need manual handling">;
def ccc_arcmt_modify : Flag<["-"], "ccc-arcmt-modify">, CCCDriverOpt,
  HelpText<"Apply modifications to files to conform to ARC">;
def ccc_arrmt_check : Flag<["-"], "ccc-arrmt-check">, Alias<ccc_arcmt_check>;
def ccc_arrmt_modify : Flag<["-"], "ccc-arrmt-modify">, Alias<ccc_arcmt_modify>;
def ccc_arcmt_migrate : Separate<["-"], "ccc-arcmt-migrate">, CCCDriverOpt,
  HelpText<"Apply modifications and produces temporary files that conform to ARC">;
def arcmt_migrate_report_output : Separate<["-"], "arcmt-migrate-report-output">,
  HelpText<"Output path for the plist report">,  Flags<[CC1Option]>;
def arcmt_migrate_emit_arc_errors : Flag<["-"], "arcmt-migrate-emit-errors">,
  HelpText<"Emit ARC errors even if the migrator can fix them">,
  Flags<[CC1Option]>;

def _migrate : Flag<["--"], "migrate">, Flags<[DriverOption]>,
  HelpText<"Run the migrator">;
def ccc_objcmt_migrate : Separate<["-"], "ccc-objcmt-migrate">, CCCDriverOpt,
  HelpText<"Apply modifications and produces temporary files to migrate to "
   "modern ObjC syntax">;
def objcmt_migrate_literals : Flag<["-"], "objcmt-migrate-literals">, Flags<[CC1Option]>,
  HelpText<"Enable migration to modern ObjC literals">;
def objcmt_migrate_subscripting : Flag<["-"], "objcmt-migrate-subscripting">, Flags<[CC1Option]>,
  HelpText<"Enable migration to modern ObjC subscripting">;

// Make sure all other -ccc- options are rejected.
def ccc_ : Joined<["-"], "ccc-">, Group<ccc_Group>, Flags<[Unsupported]>;

// Standard Options

def _HASH_HASH_HASH : Flag<["-"], "###">, Flags<[DriverOption]>,
    HelpText<"Print the commands to run for this compilation">;
// The '--' option is here for the sake of compatibility with gcc, but is 
// being ignored by the driver.
def _DASH_DASH : Flag<["--"], "">, Flags<[DriverOption]>;
def A : JoinedOrSeparate<["-"], "A">;
def B : JoinedOrSeparate<["-"], "B">;
def CC : Flag<["-"], "CC">, Flags<[CC1Option]>;
def C : Flag<["-"], "C">, Flags<[CC1Option]>;
def D : JoinedOrSeparate<["-"], "D">, Group<CompileOnly_Group>, Flags<[CC1Option]>;
def E : Flag<["-"], "E">, Flags<[DriverOption,CC1Option]>, Group<Action_Group>,
  HelpText<"Only run the preprocessor">;
def F : JoinedOrSeparate<["-"], "F">, Flags<[RenderJoined,CC1Option]>,
    HelpText<"Add directory to framework include search path">;
def G : Separate<["-"], "G">, Flags<[DriverOption]>;
def H : Flag<["-"], "H">, Flags<[CC1Option]>,
    HelpText<"Show header includes and nesting depth">;
def I_ : Flag<["-"], "I-">, Group<I_Group>;
def I : JoinedOrSeparate<["-"], "I">, Group<I_Group>, Flags<[CC1Option]>,
    HelpText<"Add directory to include search path">;
def L : JoinedOrSeparate<["-"], "L">, Flags<[RenderJoined]>;
def MD : Flag<["-"], "MD">, Group<M_Group>;
def MF : JoinedOrSeparate<["-"], "MF">, Group<M_Group>;
def MG : Flag<["-"], "MG">, Group<M_Group>, Flags<[CC1Option]>,
    HelpText<"Add missing headers to dependency list">;
def MMD : Flag<["-"], "MMD">, Group<M_Group>;
def MM : Flag<["-"], "MM">, Group<M_Group>;
def MP : Flag<["-"], "MP">, Group<M_Group>, Flags<[CC1Option]>,
    HelpText<"Create phony target for each dependency (other than main file)">;
def MQ : JoinedOrSeparate<["-"], "MQ">, Group<M_Group>, Flags<[CC1Option]>,
    HelpText<"Specify target to quote for dependency">;
def MT : JoinedOrSeparate<["-"], "MT">, Group<M_Group>, Flags<[CC1Option]>,
    HelpText<"Specify target for dependency">;
def Mach : Flag<["-"], "Mach">;
def M : Flag<["-"], "M">, Group<M_Group>;
def O0 : Joined<["-"], "O0">, Group<O_Group>, Flags<[CC1Option]>;
def O4 : Joined<["-"], "O4">, Group<O_Group>, Flags<[CC1Option]>;
def ObjCXX : Flag<["-"], "ObjC++">, Flags<[DriverOption]>,
  HelpText<"Treat source input files as Objective-C++ inputs">;
def ObjC : Flag<["-"], "ObjC">, Flags<[DriverOption]>,
  HelpText<"Treat source input files as Objective-C inputs">;
def O : Joined<["-"], "O">, Group<O_Group>, Flags<[CC1Option]>;
def P : Flag<["-"], "P">, Flags<[CC1Option]>,
  HelpText<"Disable linemarker output in -E mode">;
def Qn : Flag<["-"], "Qn">;
def Qunused_arguments : Flag<["-"], "Qunused-arguments">, Flags<[DriverOption]>,
  HelpText<"Don't emit warning for unused driver arguments">;
def Q : Flag<["-"], "Q">;
def R : Flag<["-"], "R">;
def S : Flag<["-"], "S">, Flags<[DriverOption,CC1Option]>, Group<Action_Group>,
  HelpText<"Only run preprocess and compilation steps">;
def Tbss : JoinedOrSeparate<["-"], "Tbss">, Group<T_Group>;
def Tdata : JoinedOrSeparate<["-"], "Tdata">, Group<T_Group>;
def Ttext : JoinedOrSeparate<["-"], "Ttext">, Group<T_Group>;
def T : JoinedOrSeparate<["-"], "T">, Group<T_Group>;
def U : JoinedOrSeparate<["-"], "U">, Group<CompileOnly_Group>, Flags<[CC1Option]>;
def V : JoinedOrSeparate<["-"], "V">, Flags<[DriverOption, Unsupported]>;
def Wa_COMMA : CommaJoined<["-"], "Wa,">,
  HelpText<"Pass the comma separated arguments in <arg> to the assembler">,
  MetaVarName<"<arg>">;
def Wall : Flag<["-"], "Wall">, Group<W_Group>, Flags<[CC1Option]>;
def Wdeprecated : Flag<["-"], "Wdeprecated">, Group<W_Group>, Flags<[CC1Option]>;
def Wno_deprecated : Flag<["-"], "Wno-deprecated">, Group<W_Group>, Flags<[CC1Option]>;
def Wextra : Flag<["-"], "Wextra">, Group<W_Group>, Flags<[CC1Option]>;
def Wl_COMMA : CommaJoined<["-"], "Wl,">, Flags<[LinkerInput, RenderAsInput]>,
  HelpText<"Pass the comma separated arguments in <arg> to the linker">,
  MetaVarName<"<arg>">;
def Wno_nonportable_cfstrings : Joined<["-"], "Wno-nonportable-cfstrings">, Group<W_Group>,
  Flags<[CC1Option]>;
def Wnonportable_cfstrings : Joined<["-"], "Wnonportable-cfstrings">, Group<W_Group>,
  Flags<[CC1Option]>;
def Wp_COMMA : CommaJoined<["-"], "Wp,">,
  HelpText<"Pass the comma separated arguments in <arg> to the preprocessor">,
  MetaVarName<"<arg>">;
def Wwrite_strings : Flag<["-"], "Wwrite-strings">, Group<W_Group>, Flags<[CC1Option]>;
def Wno_write_strings : Flag<["-"], "Wno-write-strings">, Group<W_Group>, Flags<[CC1Option]>;
def W_Joined : Joined<["-"], "W">, Group<W_Group>, Flags<[CC1Option]>,
  MetaVarName<"<warning>">, HelpText<"Enable the specified warning">;
def Xanalyzer : Separate<["-"], "Xanalyzer">,
  HelpText<"Pass <arg> to the static analyzer">, MetaVarName<"<arg>">;
def Xarch__ : JoinedAndSeparate<["-"], "Xarch_">, Flags<[DriverOption]>;
def Xassembler : Separate<["-"], "Xassembler">,
  HelpText<"Pass <arg> to the assembler">, MetaVarName<"<arg>">;
def Xclang : Separate<["-"], "Xclang">,
  HelpText<"Pass <arg> to the clang compiler">, MetaVarName<"<arg>">,
  Flags<[NoForward]>;
def Xlinker : Separate<["-"], "Xlinker">, Flags<[LinkerInput, RenderAsInput]>,
  HelpText<"Pass <arg> to the linker">, MetaVarName<"<arg>">;
def Xpreprocessor : Separate<["-"], "Xpreprocessor">,
  HelpText<"Pass <arg> to the preprocessor">, MetaVarName<"<arg>">;
def X_Flag : Flag<["-"], "X">;
def X_Joined : Joined<["-"], "X">;
def Z_Flag : Flag<["-"], "Z">;
def Z_Joined : Joined<["-"], "Z">;
def all__load : Flag<["-"], "all_load">;
def allowable__client : Separate<["-"], "allowable_client">;
def ansi : Flag<["-", "--"], "ansi">, Group<a_Group>;
def arch__errors__fatal : Flag<["-"], "arch_errors_fatal">;
def arch : Separate<["-"], "arch">, Flags<[DriverOption]>;
def arch__only : Separate<["-"], "arch_only">;
def a : Joined<["-"], "a">, Group<a_Group>;
def bind__at__load : Flag<["-"], "bind_at_load">;
def bundle__loader : Separate<["-"], "bundle_loader">;
def bundle : Flag<["-"], "bundle">;
def b : JoinedOrSeparate<["-"], "b">, Flags<[Unsupported]>;
def cl_kernel_arg_info : Flag<["-"], "cl-kernel-arg-info">, Flags<[CC1Option]>, Group<opencl_Group>,
HelpText<"OpenCL only. This option allows the compiler to store information about the arguments of a kernel(s)"> ;
def client__name : JoinedOrSeparate<["-"], "client_name">;
def combine : Flag<["-", "--"], "combine">, Flags<[DriverOption, Unsupported]>;
def compatibility__version : JoinedOrSeparate<["-"], "compatibility_version">;
def coverage : Flag<["-", "--"], "coverage">;
def cpp_precomp : Flag<["-"], "cpp-precomp">, Group<clang_ignored_f_Group>;
def current__version : JoinedOrSeparate<["-"], "current_version">;
def cxx_isystem : JoinedOrSeparate<["-"], "cxx-isystem">, Group<clang_i_Group>,
  HelpText<"Add directory to the C++ SYSTEM include search path">, Flags<[CC1Option]>,
  MetaVarName<"<directory>">;
def c : Flag<["-"], "c">, Flags<[DriverOption]>,
  HelpText<"Only run preprocess, compile, and assemble steps">;
def dA : Flag<["-"], "dA">, Group<d_Group>;
def dD : Flag<["-"], "dD">, Group<d_Group>, Flags<[CC1Option]>,
  HelpText<"Print macro definitions in -E mode in addition to normal output">;
def dM : Flag<["-"], "dM">, Group<d_Group>, Flags<[CC1Option]>,
  HelpText<"Print macro definitions in -E mode instead of normal output">;
def dead__strip : Flag<["-"], "dead_strip">;
def dependency_file : Separate<["-"], "dependency-file">, Flags<[CC1Option]>,
  HelpText<"Filename (or -) to write dependency output to">;
def dependency_dot : Separate<["-"], "dependency-dot">, Flags<[CC1Option]>,
  HelpText<"Filename to write DOT-formatted header dependencies to">;
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
def emit_ast : Flag<["-"], "emit-ast">,
  HelpText<"Emit Clang AST files for source inputs">;
def emit_llvm : Flag<["-"], "emit-llvm">, Flags<[CC1Option]>, Group<Action_Group>,
  HelpText<"Use the LLVM representation for assembler and object files">;
def exported__symbols__list : Separate<["-"], "exported_symbols_list">;
def e : JoinedOrSeparate<["-"], "e">;
def fPIC : Flag<["-"], "fPIC">, Group<f_Group>;
def fno_PIC : Flag<["-"], "fno-PIC">, Group<f_Group>;
def fPIE : Flag<["-"], "fPIE">, Group<f_Group>;
def fno_PIE : Flag<["-"], "fno-PIE">, Group<f_Group>;
def faccess_control : Flag<["-"], "faccess-control">, Group<f_Group>;
def fallow_unsupported : Flag<["-"], "fallow-unsupported">, Group<f_Group>;
def faltivec : Flag<["-"], "faltivec">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable AltiVec vector initializer syntax">;
def fapple_kext : Flag<["-"], "fapple-kext">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Use Apple's kernel extensions ABI">;
def fapple_pragma_pack : Flag<["-"], "fapple-pragma-pack">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable Apple gcc-compatible #pragma pack handling">;
def faddress_sanitizer : Flag<["-"], "faddress-sanitizer">, Group<f_Group>;
def fno_address_sanitizer : Flag<["-"], "fno-address-sanitizer">, Group<f_Group>;
def fthread_sanitizer : Flag<["-"], "fthread-sanitizer">, Group<f_Group>;
def fno_thread_sanitizer : Flag<["-"], "fno-thread-sanitizer">, Group<f_Group>;
def fasm : Flag<["-"], "fasm">, Group<f_Group>;

def fasm_blocks : Flag<["-"], "fasm-blocks">, Group<f_Group>;
def fno_asm_blocks : Flag<["-"], "fno-asm-blocks">, Group<f_Group>;

def fassume_sane_operator_new : Flag<["-"], "fassume-sane-operator-new">, Group<f_Group>;
def fastcp : Flag<["-"], "fastcp">, Group<f_Group>;
def fastf : Flag<["-"], "fastf">, Group<f_Group>;
def fast : Flag<["-"], "fast">, Group<f_Group>;
def fasynchronous_unwind_tables : Flag<["-"], "fasynchronous-unwind-tables">, Group<f_Group>;
def fblocks : Flag<["-"], "fblocks">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable the 'blocks' language feature">;
def fbootclasspath_EQ : Joined<["-"], "fbootclasspath=">, Group<f_Group>;
def fborland_extensions : Flag<["-"], "fborland-extensions">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Accept non-standard constructs supported by the Borland compiler">;
def fbounds_checking : Flag<["-"], "fbounds-checking">, Group<f_Group>,
  HelpText<"Enable run-time bounds checks">;
def fbounds_checking_EQ : Joined<["-"], "fbounds-checking=">, Flags<[CC1Option]>,
  Group<f_Group>;
def fbuiltin_strcat : Flag<["-"], "fbuiltin-strcat">, Group<f_Group>;
def fbuiltin_strcpy : Flag<["-"], "fbuiltin-strcpy">, Group<f_Group>;
def fbuiltin : Flag<["-"], "fbuiltin">, Group<f_Group>;
def fcaret_diagnostics : Flag<["-"], "fcaret-diagnostics">, Group<f_Group>;
def fcatch_undefined_behavior : Flag<["-"], "fcatch-undefined-behavior">, Group<f_Group>;
def fclasspath_EQ : Joined<["-"], "fclasspath=">, Group<f_Group>;
def fcolor_diagnostics : Flag<["-"], "fcolor-diagnostics">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Use colors in diagnostics">;
def fcommon : Flag<["-"], "fcommon">, Group<f_Group>;
def fcompile_resource_EQ : Joined<["-"], "fcompile-resource=">, Group<f_Group>;
def fconstant_cfstrings : Flag<["-"], "fconstant-cfstrings">, Group<f_Group>;
def fconstant_string_class_EQ : Joined<["-"], "fconstant-string-class=">, Group<f_Group>;
def fconstexpr_depth_EQ : Joined<["-"], "fconstexpr-depth=">, Group<f_Group>;
def fconstexpr_backtrace_limit_EQ : Joined<["-"], "fconstexpr-backtrace-limit=">,
                                    Group<f_Group>;
def fno_crash_diagnostics : Flag<["-"], "fno-crash-diagnostics">, Group<f_clang_Group>, Flags<[NoArgumentUnused]>;
def fcreate_profile : Flag<["-"], "fcreate-profile">, Group<f_Group>;
def fcxx_exceptions: Flag<["-"], "fcxx-exceptions">, Group<f_Group>,
  HelpText<"Enable C++ exceptions">, Flags<[CC1Option]>;
def fcxx_modules : Flag <["-"], "fcxx-modules">, Group<f_Group>, Flags<[NoForward]>;
def fdebug_pass_arguments : Flag<["-"], "fdebug-pass-arguments">, Group<f_Group>;
def fdebug_pass_structure : Flag<["-"], "fdebug-pass-structure">, Group<f_Group>;
def fdiagnostics_fixit_info : Flag<["-"], "fdiagnostics-fixit-info">, Group<f_clang_Group>;
def fdiagnostics_parseable_fixits : Flag<["-"], "fdiagnostics-parseable-fixits">, Group<f_clang_Group>,
    Flags<[CC1Option]>, HelpText<"Print fix-its in machine parseable form">;
def fdiagnostics_print_source_range_info : Flag<["-"], "fdiagnostics-print-source-range-info">,
    Group<f_clang_Group>,  Flags<[CC1Option]>,
    HelpText<"Print source range spans in numeric form">;
def fdiagnostics_show_option : Flag<["-"], "fdiagnostics-show-option">, Group<f_Group>,
    Flags<[CC1Option]>, HelpText<"Print option name with mappable diagnostics">;
def fdiagnostics_show_name : Flag<["-"], "fdiagnostics-show-name">, Group<f_Group>,
    Flags<[CC1Option]>, HelpText<"Print diagnostic name">;
def fdiagnostics_show_note_include_stack : Flag<["-"], "fdiagnostics-show-note-include-stack">,
    Group<f_Group>,  Flags<[CC1Option]>, HelpText<"Display include stacks for diagnostic notes">;
def fdiagnostics_format_EQ : Joined<["-"], "fdiagnostics-format=">, Group<f_clang_Group>;
def fdiagnostics_show_category_EQ : Joined<["-"], "fdiagnostics-show-category=">, Group<f_clang_Group>;
def fdiagnostics_show_template_tree : Flag<["-"], "fdiagnostics-show-template-tree">,
    Group<f_Group>, Flags<[CC1Option]>,
    HelpText<"Print a template comparison tree for differing templates">;
def fdollars_in_identifiers : Flag<["-"], "fdollars-in-identifiers">, Group<f_Group>,
  HelpText<"Allow '$' in identifiers">, Flags<[CC1Option]>;
def fdwarf2_cfi_asm : Flag<["-"], "fdwarf2-cfi-asm">, Group<f_Group>;
def fno_dwarf2_cfi_asm : Flag<["-"], "fno-dwarf2-cfi-asm">, Group<f_Group>,  Flags<[CC1Option]>;
def fdwarf_directory_asm : Flag<["-"], "fdwarf-directory-asm">, Group<f_Group>;
def fno_dwarf_directory_asm : Flag<["-"], "fno-dwarf-directory-asm">, Group<f_Group>, Flags<[CC1Option]>;
def felide_constructors : Flag<["-"], "felide-constructors">, Group<f_Group>;
def fno_elide_type : Flag<["-"], "fno-elide-type">, Group<f_Group>,
    Flags<[CC1Option]>,
    HelpText<"Do not elide types when printing diagnostics">;
def feliminate_unused_debug_symbols : Flag<["-"], "feliminate-unused-debug-symbols">, Group<f_Group>;
def femit_all_decls : Flag<["-"], "femit-all-decls">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Emit all declarations, even if unused">;
def fencoding_EQ : Joined<["-"], "fencoding=">, Group<f_Group>;
def ferror_limit_EQ : Joined<["-"], "ferror-limit=">, Group<f_Group>;
def fexceptions : Flag<["-"], "fexceptions">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable support for exception handling">;
def fextdirs_EQ : Joined<["-"], "fextdirs=">, Group<f_Group>;
def fhosted : Flag<["-"], "fhosted">, Group<f_Group>;
def ffast_math : Flag<["-"], "ffast-math">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable the *frontend*'s 'fast-math' mode. This has no effect on "
           "optimizations, but provides a preprocessor macro __FAST_MATH__ the "
           "same as GCC's -ffast-math flag">;
def fno_fast_math : Flag<["-"], "fno-fast-math">, Group<f_Group>;
def fmath_errno : Flag<["-"], "fmath-errno">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Require math functions to indicate errors by setting errno">;
def fno_math_errno : Flag<["-"], "fno-math-errno">, Group<f_Group>;
def fsignaling_math : Flag<["-"], "fsignaling-math">, Group<f_Group>;
def fno_signaling_math : Flag<["-"], "fno-signaling-math">, Group<f_Group>;
def fsanitize_EQ : CommaJoined<["-"], "fsanitize=">, Group<f_clang_Group>,
                   Flags<[CC1Option]>, MetaVarName<"<check>">,
                   HelpText<"Enable runtime instrumentation for bug detection: "
                            "address (memory errors) | thread (race detection) | "
                            "undefined (miscellaneous undefined behavior)">;
def fno_sanitize_EQ : CommaJoined<["-"], "fno-sanitize=">, Group<f_clang_Group>;
def funsafe_math_optimizations : Flag<["-"], "funsafe-math-optimizations">,
  Group<f_Group>;
def fno_unsafe_math_optimizations : Flag<["-"], "fno-unsafe-math-optimizations">,
  Group<f_Group>;
def fassociative_math : Flag<["-"], "fassociative-math">, Group<f_Group>;
def fno_associative_math : Flag<["-"], "fno-associative-math">, Group<f_Group>;
def freciprocal_math : Flag<["-"], "freciprocal-math">, Group<f_Group>;
def fno_reciprocal_math : Flag<["-"], "fno-reciprocal-math">, Group<f_Group>;
def ffinite_math_only : Flag<["-"], "ffinite-math-only">, Group<f_Group>, Flags<[CC1Option]>;
def fno_finite_math_only : Flag<["-"], "fno-finite-math-only">, Group<f_Group>;
def fsigned_zeros : Flag<["-"], "fsigned-zeros">, Group<f_Group>;
def fno_signed_zeros : Flag<["-"], "fno-signed-zeros">, Group<f_Group>;
def fhonor_nans : Flag<["-"], "fhonor-nans">, Group<f_Group>;
def fno_honor_nans : Flag<["-"], "fno-honor-nans">, Group<f_Group>;
def fhonor_infinities : Flag<["-"], "fhonor-infinities">, Group<f_Group>;
def fno_honor_infinities : Flag<["-"], "fno-honor-infinities">, Group<f_Group>;
// Sic. This option was misspelled originally.
def fhonor_infinites : Flag<["-"], "fhonor-infinites">, Alias<fhonor_infinities>;
def fno_honor_infinites : Flag<["-"], "fno-honor-infinites">, Alias<fno_honor_infinities>;
def ftrapping_math : Flag<["-"], "ftrapping-math">, Group<f_Group>;
def fno_trapping_math : Flag<["-"], "fno-trapping-math">, Group<f_Group>;
def ffp_contract : Joined<["-"], "ffp-contract=">, Group<f_Group>,
  Flags<[CC1Option]>, HelpText<"Form fused FP ops (e.g. FMAs): fast (everywhere)"
  " | on (according to FP_CONTRACT pragma, default) | off (never fuse)">;

def ffor_scope : Flag<["-"], "ffor-scope">, Group<f_Group>;
def fno_for_scope : Flag<["-"], "fno-for-scope">, Group<f_Group>;

def frewrite_includes : Flag<["-"], "frewrite-includes">, Group<f_Group>,
  Flags<[CC1Option]>;
def fno_rewrite_includes : Flag<["-"], "fno-rewrite-includes">, Group<f_Group>;

def ffreestanding : Flag<["-"], "ffreestanding">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Assert that the compilation takes place in a freestanding environment">;
def fgnu_keywords : Flag<["-"], "fgnu-keywords">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Allow GNU-extension keywords regardless of language standard">;
def fgnu89_inline : Flag<["-"], "fgnu89-inline">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Use the gnu89 inline semantics">;
def fno_gnu89_inline : Flag<["-"], "fno-gnu89-inline">, Group<f_Group>;
def fgnu_runtime : Flag<["-"], "fgnu-runtime">, Group<f_Group>,
  HelpText<"Generate output compatible with the standard GNU Objective-C runtime">;
def fheinous_gnu_extensions : Flag<["-"], "fheinous-gnu-extensions">, Flags<[CC1Option]>;
def filelist : Separate<["-"], "filelist">, Flags<[LinkerInput]>;
def findirect_virtual_calls : Flag<["-"], "findirect-virtual-calls">, Alias<fapple_kext>;
def finline_functions : Flag<["-"], "finline-functions">, Group<clang_ignored_f_Group>;
def finline : Flag<["-"], "finline">, Group<clang_ignored_f_Group>;
def finstrument_functions : Flag<["-"], "finstrument-functions">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Generate calls to instrument function entry and exit">;
def fkeep_inline_functions : Flag<["-"], "fkeep-inline-functions">, Group<clang_ignored_f_Group>;
def flat__namespace : Flag<["-"], "flat_namespace">;
def flax_vector_conversions : Flag<["-"], "flax-vector-conversions">, Group<f_Group>;
def flimit_debug_info : Flag<["-"], "flimit-debug-info">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Limit debug information produced to reduce size of debug binary">;
def flimited_precision_EQ : Joined<["-"], "flimited-precision=">, Group<f_Group>;
def flto : Flag<["-"], "flto">, Group<f_Group>;
def fno_lto : Flag<["-"], "fno-lto">, Group<f_Group>;
def fmacro_backtrace_limit_EQ : Joined<["-"], "fmacro-backtrace-limit=">,
                                Group<f_Group>;
def fmerge_all_constants : Flag<["-"], "fmerge-all-constants">, Group<f_Group>;
def fmessage_length_EQ : Joined<["-"], "fmessage-length=">, Group<f_Group>;
def fms_extensions : Flag<["-"], "fms-extensions">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Accept some non-standard constructs supported by the Microsoft compiler">;
def fenable_experimental_ms_inline_asm : Flag<["-"], "fenable-experimental-ms-inline-asm">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable support for Microsoft style inine assembly">;
def fms_compatibility : Flag<["-"], "fms-compatibility">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable Microsoft compatibility mode">;
def fmsc_version : Joined<["-"], "fmsc-version=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Version of the Microsoft C/C++ compiler to report in _MSC_VER (0 = don't define it (default))">;
def fdelayed_template_parsing : Flag<["-"], "fdelayed-template-parsing">, Group<f_Group>,
  HelpText<"Parse templated function definitions at the end of the "
           "translation unit ">,  Flags<[CC1Option]>;
def fmodule_cache_path : Separate<["-"], "fmodule-cache-path">, Group<i_Group>, 
  Flags<[NoForward,CC1Option]>, MetaVarName<"<directory>">,
  HelpText<"Specify the module cache path">;
def fmodules : Flag <["-"], "fmodules">, Group<f_Group>, Flags<[NoForward,CC1Option]>,
  HelpText<"Enable the 'modules' language feature">;
def fretain_comments_from_system_headers : Flag<["-"], "fretain-comments-from-system-headers">, Group<f_Group>, Flags<[CC1Option]>;

def fmudflapth : Flag<["-"], "fmudflapth">, Group<f_Group>;
def fmudflap : Flag<["-"], "fmudflap">, Group<f_Group>;
def fnested_functions : Flag<["-"], "fnested-functions">, Group<f_Group>;
def fnext_runtime : Flag<["-"], "fnext-runtime">, Group<f_Group>;
def fno_access_control : Flag<["-"], "fno-access-control">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Disable C++ access control">;
def fno_apple_pragma_pack : Flag<["-"], "fno-apple-pragma-pack">, Group<f_Group>;
def fno_asm : Flag<["-"], "fno-asm">, Group<f_Group>;
def fno_asynchronous_unwind_tables : Flag<["-"], "fno-asynchronous-unwind-tables">, Group<f_Group>;
def fno_assume_sane_operator_new : Flag<["-"], "fno-assume-sane-operator-new">, Group<f_Group>,
  HelpText<"Don't assume that C++'s global operator new can't alias any pointer">,
  Flags<[CC1Option]>;
def fno_blocks : Flag<["-"], "fno-blocks">, Group<f_Group>;
def fno_borland_extensions : Flag<["-"], "fno-borland-extensions">, Group<f_Group>;
def fno_builtin_strcat : Flag<["-"], "fno-builtin-strcat">, Group<f_Group>;
def fno_builtin_strcpy : Flag<["-"], "fno-builtin-strcpy">, Group<f_Group>;
def fno_builtin : Flag<["-"], "fno-builtin">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Disable implicit builtin knowledge of functions">;
def fno_caret_diagnostics : Flag<["-"], "fno-caret-diagnostics">, Group<f_Group>,
 Flags<[CC1Option]>;
def fno_color_diagnostics : Flag<["-"], "fno-color-diagnostics">, Group<f_Group>;
def fno_common : Flag<["-"], "fno-common">, Group<f_Group>, Flags<[CC1Option]>,
    HelpText<"Compile common globals like normal definitions">;
def fno_constant_cfstrings : Flag<["-"], "fno-constant-cfstrings">, Group<f_Group>,
  Flags<[CC1Option]>,
  HelpText<"Disable creation of CodeFoundation-type constant strings">;
def fno_cxx_exceptions: Flag<["-"], "fno-cxx-exceptions">, Group<f_Group>;
def fno_cxx_modules : Flag <["-"], "fno-cxx-modules">, Group<f_Group>, Flags<[NoForward]>;
def fno_diagnostics_fixit_info : Flag<["-"], "fno-diagnostics-fixit-info">, Group<f_Group>,
  Flags<[CC1Option]>, HelpText<"Do not include fixit information in diagnostics">;
def fno_diagnostics_show_name : Flag<["-"], "fno-diagnostics-show-name">, Group<f_Group>;
def fno_diagnostics_show_option : Flag<["-"], "fno-diagnostics-show-option">, Group<f_Group>;
def fno_diagnostics_show_note_include_stack : Flag<["-"], "fno-diagnostics-show-note-include-stack">,
    Flags<[CC1Option]>, Group<f_Group>, HelpText<"Display include stacks for diagnostic notes">;
def fno_dollars_in_identifiers : Flag<["-"], "fno-dollars-in-identifiers">, Group<f_Group>,
  HelpText<"Disallow '$' in identifiers">, Flags<[CC1Option]>;
def fno_elide_constructors : Flag<["-"], "fno-elide-constructors">, Group<f_Group>,
  HelpText<"Disable C++ copy constructor elision">, Flags<[CC1Option]>;
def fno_eliminate_unused_debug_symbols : Flag<["-"], "fno-eliminate-unused-debug-symbols">, Group<f_Group>;
def fno_exceptions : Flag<["-"], "fno-exceptions">, Group<f_Group>;
def fno_gnu_keywords : Flag<["-"], "fno-gnu-keywords">, Group<f_Group>, Flags<[CC1Option]>;
def fno_inline_functions : Flag<["-"], "fno-inline-functions">, Group<f_clang_Group>, Flags<[CC1Option]>;
def fno_inline : Flag<["-"], "fno-inline">, Group<f_clang_Group>, Flags<[CC1Option]>;
def fno_keep_inline_functions : Flag<["-"], "fno-keep-inline-functions">, Group<clang_ignored_f_Group>;
def fno_lax_vector_conversions : Flag<["-"], "fno-lax-vector-conversions">, Group<f_Group>,
  HelpText<"Disallow implicit conversions between vectors with a different number of elements or different element types">, Flags<[CC1Option]>;
def fno_limit_debug_info : Flag<["-"], "fno-limit-debug-info">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Do not limit debug information produced to reduce size of debug binary">;
def fno_merge_all_constants : Flag<["-"], "fno-merge-all-constants">, Group<f_Group>,
    Flags<[CC1Option]>, HelpText<"Disallow merging of constants">;
def fno_modules : Flag <["-"], "fno-modules">, Group<f_Group>, Flags<[NoForward]>;
def fno_ms_extensions : Flag<["-"], "fno-ms-extensions">, Group<f_Group>;
def fno_ms_compatibility : Flag<["-"], "fno-ms-compatibility">, Group<f_Group>;
def fno_delayed_template_parsing : Flag<["-"], "fno-delayed-template-parsing">, Group<f_Group>;
def fno_objc_exceptions: Flag<["-"], "fno-objc-exceptions">, Group<f_Group>;
def fno_objc_legacy_dispatch : Flag<["-"], "fno-objc-legacy-dispatch">, Group<f_Group>;
def fno_omit_frame_pointer : Flag<["-"], "fno-omit-frame-pointer">, Group<f_Group>;
def fno_operator_names : Flag<["-"], "fno-operator-names">, Group<f_Group>,
  HelpText<"Do not treat C++ operator name keywords as synonyms for operators">,
  Flags<[CC1Option]>;
def fno_pascal_strings : Flag<["-"], "fno-pascal-strings">, Group<f_Group>;
def fno_rtti : Flag<["-"], "fno-rtti">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Disable generation of rtti information">;
def fno_short_enums : Flag<["-"], "fno-short-enums">, Group<f_Group>;
def fno_show_column : Flag<["-"], "fno-show-column">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Do not include column number on diagnostics">;
def fno_show_source_location : Flag<["-"], "fno-show-source-location">, Group<f_Group>,
  Flags<[CC1Option]>, HelpText<"Do not include source location information with diagnostics">;
def fno_spell_checking : Flag<["-"], "fno-spell-checking">, Group<f_Group>,
  Flags<[CC1Option]>, HelpText<"Disable spell-checking">;
def fno_stack_protector : Flag<["-"], "fno-stack-protector">, Group<f_Group>;
def fno_strict_aliasing : Flag<["-"], "fno-strict-aliasing">, Group<f_Group>;
def fno_strict_enums : Flag<["-"], "fno-strict-enums">, Group<f_Group>;
def fno_strict_overflow : Flag<["-"], "fno-strict-overflow">, Group<f_Group>;
def fno_threadsafe_statics : Flag<["-"], "fno-threadsafe-statics">, Group<f_Group>,
  Flags<[CC1Option]>, HelpText<"Do not emit code to make initialization of local statics thread safe">;
def fno_use_cxa_atexit : Flag<["-"], "fno-use-cxa-atexit">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Don't use __cxa_atexit for calling destructors">;
def fno_unit_at_a_time : Flag<["-"], "fno-unit-at-a-time">, Group<f_Group>;
def fno_unwind_tables : Flag<["-"], "fno-unwind-tables">, Group<f_Group>;
def fno_verbose_asm : Flag<["-"], "fno-verbose-asm">, Group<f_Group>;
def fno_working_directory : Flag<["-"], "fno-working-directory">, Group<f_Group>;
def fno_wrapv : Flag<["-"], "fno-wrapv">, Group<f_Group>;
def fno_zero_initialized_in_bss : Flag<["-"], "fno-zero-initialized-in-bss">, Group<f_Group>;
def fobjc_arc : Flag<["-"], "fobjc-arc">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Synthesize retain and release calls for Objective-C pointers">;
def fno_objc_arc : Flag<["-"], "fno-objc-arc">, Group<f_Group>;
def fobjc_arc_exceptions : Flag<["-"], "fobjc-arc-exceptions">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Use EH-safe code when synthesizing retains and releases in -fobjc-arc">;
def fno_objc_arc_exceptions : Flag<["-"], "fno-objc-arc-exceptions">, Group<f_Group>;
def fobjc_atdefs : Flag<["-"], "fobjc-atdefs">, Group<clang_ignored_f_Group>;
def fobjc_call_cxx_cdtors : Flag<["-"], "fobjc-call-cxx-cdtors">, Group<clang_ignored_f_Group>;
def fobjc_exceptions: Flag<["-"], "fobjc-exceptions">, Group<f_Group>,
  HelpText<"Enable Objective-C exceptions">, Flags<[CC1Option]>;

def fobjc_gc_only : Flag<["-"], "fobjc-gc-only">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Use GC exclusively for Objective-C related memory management">;
def fobjc_gc : Flag<["-"], "fobjc-gc">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable Objective-C garbage collection">;
def fobjc_legacy_dispatch : Flag<["-"], "fobjc-legacy-dispatch">, Group<f_Group>;
def fobjc_new_property : Flag<["-"], "fobjc-new-property">, Group<clang_ignored_f_Group>;
def fobjc_infer_related_result_type : Flag<["-"], "fobjc-infer-related-result-type">, 
                                      Group<f_Group>;
def fno_objc_infer_related_result_type : Flag<["-"],
  "fno-objc-infer-related-result-type">, Group<f_Group>,
  HelpText<
    "do not infer Objective-C related result type based on method family">,
  Flags<[CC1Option]>;
def fobjc_link_runtime: Flag<["-"], "fobjc-link-runtime">, Group<f_Group>;

// Objective-C ABI options.
def fobjc_runtime_EQ : Joined<["-"], "fobjc-runtime=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Specify the target Objective-C runtime kind and version">;
def fobjc_abi_version_EQ : Joined<["-"], "fobjc-abi-version=">, Group<f_Group>;
def fobjc_nonfragile_abi_version_EQ : Joined<["-"], "fobjc-nonfragile-abi-version=">, Group<f_Group>;
def fobjc_nonfragile_abi : Flag<["-"], "fobjc-nonfragile-abi">, Group<f_Group>;
def fno_objc_nonfragile_abi : Flag<["-"], "fno-objc-nonfragile-abi">, Group<f_Group>;

def fobjc_sender_dependent_dispatch : Flag<["-"], "fobjc-sender-dependent-dispatch">, Group<f_Group>;
def fobjc : Flag<["-"], "fobjc">, Group<f_Group>;
def fomit_frame_pointer : Flag<["-"], "fomit-frame-pointer">, Group<f_Group>;
def fopenmp : Flag<["-"], "fopenmp">, Group<f_Group>;
def fno_optimize_sibling_calls : Flag<["-"], "fno-optimize-sibling-calls">, Group<f_Group>;
def foptimize_sibling_calls : Flag<["-"], "foptimize-sibling-calls">, Group<f_Group>;
def force__cpusubtype__ALL : Flag<["-"], "force_cpusubtype_ALL">;
def force__flat__namespace : Flag<["-"], "force_flat_namespace">;
def force__load : Separate<["-"], "force_load">;
def foutput_class_dir_EQ : Joined<["-"], "foutput-class-dir=">, Group<f_Group>;
def fpack_struct : Flag<["-"], "fpack-struct">, Group<f_Group>;
def fno_pack_struct : Flag<["-"], "fno-pack-struct">, Group<f_Group>;
def fpack_struct_EQ : Joined<["-"], "fpack-struct=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Specify the default maximum struct packing alignment">;
def fpascal_strings : Flag<["-"], "fpascal-strings">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Recognize and construct Pascal-style string literals">;
def fpch_preprocess : Flag<["-"], "fpch-preprocess">, Group<f_Group>;
def fpic : Flag<["-"], "fpic">, Group<f_Group>;
def fno_pic : Flag<["-"], "fno-pic">, Group<f_Group>;
def fpie : Flag<["-"], "fpie">, Group<f_Group>;
def fno_pie : Flag<["-"], "fno-pie">, Group<f_Group>;
def fprofile_arcs : Flag<["-"], "fprofile-arcs">, Group<f_Group>;
def fprofile_generate : Flag<["-"], "fprofile-generate">, Group<f_Group>;
def framework : Separate<["-"], "framework">, Flags<[LinkerInput]>;
def frandom_seed_EQ : Joined<["-"], "frandom-seed=">, Group<clang_ignored_f_Group>;
def frtti : Flag<["-"], "frtti">, Group<f_Group>;
def fsched_interblock : Flag<["-"], "fsched-interblock">, Group<clang_ignored_f_Group>;
def fshort_enums : Flag<["-"], "fshort-enums">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Allocate to an enum type only as many bytes as it needs for the declared range of possible values">;
def freorder_blocks : Flag<["-"], "freorder-blocks">, Group<clang_ignored_f_Group>;
def fshort_wchar : Flag<["-"], "fshort-wchar">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Force wchar_t to be a short unsigned int">;
def fshow_overloads_EQ : Joined<["-"], "fshow-overloads=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Which overload candidates to show when overload resolution fails: "
           "best|all; defaults to all">;
def fshow_column : Flag<["-"], "fshow-column">, Group<f_Group>, Flags<[CC1Option]>;
def fshow_source_location : Flag<["-"], "fshow-source-location">, Group<f_Group>;
def fspell_checking : Flag<["-"], "fspell-checking">, Group<f_Group>;
def fsigned_bitfields : Flag<["-"], "fsigned-bitfields">, Group<f_Group>;
def fsigned_char : Flag<["-"], "fsigned-char">, Group<f_Group>;
def fstack_protector_all : Flag<["-"], "fstack-protector-all">, Group<f_Group>;
def fstack_protector : Flag<["-"], "fstack-protector">, Group<f_Group>;
def fstrict_aliasing : Flag<["-"], "fstrict-aliasing">, Group<f_Group>;
def fstrict_enums : Flag<["-"], "fstrict-enums">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Enable optimizations based on the strict definition of an enum's "
           "value range">;
def fstrict_overflow : Flag<["-"], "fstrict-overflow">, Group<f_Group>;
def fsyntax_only : Flag<["-"], "fsyntax-only">, Flags<[DriverOption,CC1Option]>, Group<Action_Group>;
def ftabstop_EQ : Joined<["-"], "ftabstop=">, Group<f_Group>;
def ftemplate_depth_EQ : Joined<["-"], "ftemplate-depth=">, Group<f_Group>;
def ftemplate_depth_ : Joined<["-"], "ftemplate-depth-">, Group<f_Group>;
def ftemplate_backtrace_limit_EQ : Joined<["-"], "ftemplate-backtrace-limit=">,
                                   Group<f_Group>;
def ftest_coverage : Flag<["-"], "ftest-coverage">, Group<f_Group>;
def Wlarge_by_value_copy_def : Flag<["-"], "Wlarge-by-value-copy">,
  HelpText<"Warn if a function definition returns or accepts an object larger "
           "in bytes than a given value">, Flags<[HelpHidden]>;
def Wlarge_by_value_copy_EQ : Joined<["-"], "Wlarge-by-value-copy=">, Flags<[CC1Option]>;

// Just silence warnings about -Wlarger-than,  -Wframe-larger-than for now.
def Wlarger_than : Separate<["-"], "Wlarger-than">, Group<clang_ignored_f_Group>;
def Wlarger_than_EQ : Joined<["-"], "Wlarger-than=">, Alias<Wlarger_than>;
def Wlarger_than_ : Joined<["-"], "Wlarger-than-">, Alias<Wlarger_than>;
def Wframe_larger_than : Separate<["-"], "Wframe-larger-than">, Group<clang_ignored_f_Group>;
def Wframe_larger_than_EQ : Joined<["-"], "Wframe-larger-than=">, Alias<Wframe_larger_than>;

def fterminated_vtables : Flag<["-"], "fterminated-vtables">, Alias<fapple_kext>;
def fthreadsafe_statics : Flag<["-"], "fthreadsafe-statics">, Group<f_Group>;
def ftime_report : Flag<["-"], "ftime-report">, Group<f_Group>, Flags<[CC1Option]>;
def ftlsmodel_EQ : Joined<["-"], "ftls-model=">, Group<f_Group>, Flags<[CC1Option]>;
def ftrapv : Flag<["-"], "ftrapv">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Trap on integer overflow">;
def ftrapv_handler_EQ : Joined<["-"], "ftrapv-handler=">, Group<f_Group>,
  MetaVarName<"<function name>">,
  HelpText<"Specify the function to be called on overflow">;
def ftrapv_handler : Separate<["-"], "ftrapv-handler">, Group<f_Group>, Flags<[CC1Option]>;
def ftrap_function_EQ : Joined<["-"], "ftrap-function=">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Issue call to specified function rather than a trap instruction">;
def funit_at_a_time : Flag<["-"], "funit-at-a-time">, Group<f_Group>;
def funroll_loops : Flag<["-"], "funroll-loops">, Group<f_Group>,
  HelpText<"Turn on loop unroller">, Flags<[CC1Option]>;
def funsigned_bitfields : Flag<["-"], "funsigned-bitfields">, Group<f_Group>;
def funsigned_char : Flag<["-"], "funsigned-char">, Group<f_Group>;
def funwind_tables : Flag<["-"], "funwind-tables">, Group<f_Group>;
def fuse_cxa_atexit : Flag<["-"], "fuse-cxa-atexit">, Group<f_Group>;
def fverbose_asm : Flag<["-"], "fverbose-asm">, Group<f_Group>;
def fvisibility_EQ : Joined<["-"], "fvisibility=">, Group<f_Group>;
def fvisibility_inlines_hidden : Flag<["-"], "fvisibility-inlines-hidden">, Group<f_Group>,
  HelpText<"Give inline C++ member functions default visibility by default">,
  Flags<[CC1Option]>;
def fwrapv : Flag<["-"], "fwrapv">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Treat signed integer overflow as two's complement">;
def fwritable_strings : Flag<["-"], "fwritable-strings">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Store string literals as writable data">;
def fzero_initialized_in_bss : Flag<["-"], "fzero-initialized-in-bss">, Group<f_Group>;
def ffunction_sections: Flag <["-"], "ffunction-sections">, Group<f_Group>,
  Flags<[CC1Option]>, HelpText<"Place each function in its own section (ELF Only)">;
def fdata_sections : Flag <["-"], "fdata-sections">, Group<f_Group>, Flags<[CC1Option]>,
  HelpText<"Place each data in its own section (ELF Only)">;
def f : Joined<["-"], "f">, Group<f_Group>;
def g_Flag : Flag<["-"], "g">, Group<g_Group>,
  HelpText<"Generate source level debug information">, Flags<[CC1Option]>;
def gline_tables_only : Flag<["-"], "gline-tables-only">, Group<g_Group>,
  HelpText<"Emit debug line number tables only">, Flags<[CC1Option]>;
def g0 : Flag<["-"], "g0">, Group<g_Group>;
def g1 : Flag<["-"], "g1">, Group<g_Group>;
def g2 : Flag<["-"], "g2">, Group<g_Group>;
def g3 : Flag<["-"], "g3">, Group<g_Group>;
def ggdb : Flag<["-"], "ggdb">, Group<g_Group>;
def ggdb0 : Flag<["-"], "ggdb0">, Group<g_Group>;
def ggdb1 : Flag<["-"], "ggdb1">, Group<g_Group>;
def ggdb2 : Flag<["-"], "ggdb2">, Group<g_Group>;
def ggdb3 : Flag<["-"], "ggdb3">, Group<g_Group>;
def gdwarf_2 : Flag<["-"], "gdwarf-2">, Group<g_Group>;
def gdwarf_3 : Flag<["-"], "gdwarf-3">, Group<g_Group>;
def gdwarf_4 : Flag<["-"], "gdwarf-4">, Group<g_Group>;
def gfull : Flag<["-"], "gfull">, Group<g_Group>;
def gused : Flag<["-"], "gused">, Group<g_Group>;
def gstabs : Joined<["-"], "gstabs">, Group<g_Group>, Flags<[Unsupported]>;
def gcoff : Joined<["-"], "gcoff">, Group<g_Group>, Flags<[Unsupported]>;
def gxcoff : Joined<["-"], "gxcoff">, Group<g_Group>, Flags<[Unsupported]>;
def gvms : Joined<["-"], "gvms">, Group<g_Group>, Flags<[Unsupported]>;
def gtoggle : Flag<["-"], "gtoggle">, Group<g_flags_Group>, Flags<[Unsupported]>;
def grecord_gcc_switches : Flag<["-"], "grecord-gcc-switches">, Group<g_flags_Group>;
def gno_record_gcc_switches : Flag<["-"], "gno-record-gcc-switches">,
  Group<g_flags_Group>;
def gstrict_dwarf : Flag<["-"], "gstrict-dwarf">, Group<g_flags_Group>;
def gno_strict_dwarf : Flag<["-"], "gno-strict-dwarf">, Group<g_flags_Group>;
def gcolumn_info : Flag<["-"], "gcolumn-info">, Group<g_flags_Group>;
def headerpad__max__install__names : Joined<["-"], "headerpad_max_install_names">;
def help : Flag<["-", "--"], "help">, Flags<[CC1Option]>,
  HelpText<"Display available options">;
def index_header_map : Flag<["-"], "index-header-map">, Flags<[CC1Option]>,
  HelpText<"Make the next included directory (-I or -F) an indexer header map">;
def idirafter : JoinedOrSeparate<["-"], "idirafter">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Add directory to AFTER include search path">;
def iframework : JoinedOrSeparate<["-"], "iframework">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Add directory to SYSTEM framework search path">;
def imacros : JoinedOrSeparate<["-", "--"], "imacros">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Include macros from file before parsing">, MetaVarName<"<file>">;
def image__base : Separate<["-"], "image_base">;
def include_ : JoinedOrSeparate<["-", "--"], "include">, Group<clang_i_Group>, EnumName<"include">,
    MetaVarName<"<file>">, HelpText<"Include file before parsing">, Flags<[CC1Option]>;
def include_pch : Separate<["-"], "include-pch">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Include precompiled header file">, MetaVarName<"<file>">;
def init : Separate<["-"], "init">;
def install__name : Separate<["-"], "install_name">;
def integrated_as : Flag<["-"], "integrated-as">, Flags<[DriverOption]>;
def iprefix : JoinedOrSeparate<["-"], "iprefix">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Set the -iwithprefix/-iwithprefixbefore prefix">, MetaVarName<"<dir>">;
def iquote : JoinedOrSeparate<["-"], "iquote">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Add directory to QUOTE include search path">, MetaVarName<"<directory>">;
def isysroot : JoinedOrSeparate<["-"], "isysroot">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Set the system root directory (usually /)">, MetaVarName<"<dir>">;
def isystem : JoinedOrSeparate<["-"], "isystem">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Add directory to SYSTEM include search path">, MetaVarName<"<directory>">;
def iwithprefixbefore : JoinedOrSeparate<["-"], "iwithprefixbefore">, Group<clang_i_Group>,
  HelpText<"Set directory to include search path with prefix">, MetaVarName<"<dir>">,
  Flags<[CC1Option]>;
def iwithprefix : JoinedOrSeparate<["-"], "iwithprefix">, Group<clang_i_Group>, Flags<[CC1Option]>,
  HelpText<"Set directory to SYSTEM include search path with prefix">, MetaVarName<"<dir>">;
def iwithsysroot : JoinedOrSeparate<["-"], "iwithsysroot">, Group<clang_i_Group>,
  HelpText<"Add directory to SYSTEM include search path, "
           "absolute paths are relative to -isysroot">, MetaVarName<"<directory>">,
  Flags<[CC1Option]>;
def i : Joined<["-"], "i">, Group<i_Group>;
def keep__private__externs : Flag<["-"], "keep_private_externs">;
def l : JoinedOrSeparate<["-"], "l">, Flags<[LinkerInput, RenderJoined]>;
def lazy__framework : Separate<["-"], "lazy_framework">, Flags<[LinkerInput]>;
def lazy__library : Separate<["-"], "lazy_library">, Flags<[LinkerInput]>;
def m32 : Flag<["-"], "m32">, Group<m_Group>, Flags<[DriverOption]>;
def mqdsp6_compat : Flag<["-"], "mqdsp6-compat">, Group<m_Group>, Flags<[DriverOption,CC1Option]>,
  HelpText<"Enable hexagon-qdsp6 backward compatibility">;
def m3dnowa : Flag<["-"], "m3dnowa">, Group<m_x86_Features_Group>;
def m3dnow : Flag<["-"], "m3dnow">, Group<m_x86_Features_Group>;
def m64 : Flag<["-"], "m64">, Group<m_Group>, Flags<[DriverOption]>;
def mabi_EQ : Joined<["-"], "mabi=">, Group<m_Group>;
def march_EQ : Joined<["-"], "march=">, Group<m_Group>;
def maltivec : Flag<["-"], "maltivec">, Alias<faltivec>;
def mcmodel_EQ : Joined<["-"], "mcmodel=">, Group<m_Group>;
def mconstant_cfstrings : Flag<["-"], "mconstant-cfstrings">, Group<clang_ignored_m_Group>;
def mcpu_EQ : Joined<["-"], "mcpu=">, Group<m_Group>;
def mdynamic_no_pic : Joined<["-"], "mdynamic-no-pic">, Group<m_Group>;
def mfix_and_continue : Flag<["-"], "mfix-and-continue">, Group<clang_ignored_m_Group>;
def mfloat_abi_EQ : Joined<["-"], "mfloat-abi=">, Group<m_Group>;
def mfpmath_EQ : Joined<["-"], "mfpmath=">, Group<m_Group>;
def mfpu_EQ : Joined<["-"], "mfpu=">, Group<m_Group>;
def mglobal_merge : Flag<["-"], "mglobal-merge">, Group<m_Group>;
def mhard_float : Flag<["-"], "mhard-float">, Group<m_Group>;
def miphoneos_version_min_EQ : Joined<["-"], "miphoneos-version-min=">, Group<m_Group>;
def mios_version_min_EQ : Joined<["-"], "mios-version-min=">, Alias<miphoneos_version_min_EQ>;
def mios_simulator_version_min_EQ : Joined<["-"], "mios-simulator-version-min=">, Group<m_Group>;
def mkernel : Flag<["-"], "mkernel">, Group<m_Group>;
def mlinker_version_EQ : Joined<["-"], "mlinker-version=">, Flags<[NoForward]>;
def mllvm : Separate<["-"], "mllvm">, Flags<[CC1Option]>,
  HelpText<"Additional arguments to forward to LLVM's option processing">;
def mmacosx_version_min_EQ : Joined<["-"], "mmacosx-version-min=">, Group<m_Group>;
def mms_bitfields : Flag<["-"], "mms-bitfields">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Set the default structure layout to be compatible with the Microsoft compiler standard">;
def mstackrealign : Flag<["-"], "mstackrealign">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Force realign the stack at entry to every function">;
def mstack_alignment : Joined<["-"], "mstack-alignment=">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Set the stack alignment">;
def mstrict_align : Flag<["-"], "mstrict-align">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Force all memory accesses to be aligned (ARM only)">;
def mmmx : Flag<["-"], "mmmx">, Group<m_x86_Features_Group>;
def mno_3dnowa : Flag<["-"], "mno-3dnowa">, Group<m_x86_Features_Group>;
def mno_3dnow : Flag<["-"], "mno-3dnow">, Group<m_x86_Features_Group>;
def mno_constant_cfstrings : Flag<["-"], "mno-constant-cfstrings">, Group<m_Group>;
def mno_global_merge : Flag<["-"], "mno-global-merge">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Disable merging of globals">;
def mno_mmx : Flag<["-"], "mno-mmx">, Group<m_x86_Features_Group>;
def mno_pascal_strings : Flag<["-"], "mno-pascal-strings">, Group<m_Group>;
def mno_red_zone : Flag<["-"], "mno-red-zone">, Group<m_Group>;
def mno_relax_all : Flag<["-"], "mno-relax-all">, Group<m_Group>;
def mno_rtd: Flag<["-"], "mno-rtd">, Group<m_Group>;
def mno_soft_float : Flag<["-"], "mno-soft-float">, Group<m_Group>;
def mno_stackrealign : Flag<["-"], "mno-stackrealign">, Group<m_Group>;
def mno_sse2 : Flag<["-"], "mno-sse2">, Group<m_x86_Features_Group>;
def mno_sse3 : Flag<["-"], "mno-sse3">, Group<m_x86_Features_Group>;
def mno_sse4a : Flag<["-"], "mno-sse4a">, Group<m_x86_Features_Group>;
def mno_sse4 : Flag<["-"], "mno-sse4">, Group<m_x86_Features_Group>;
def mno_sse4_1 : Flag<["-"], "mno-sse4.1">, Group<m_x86_Features_Group>;
def mno_sse4_2 : Flag<["-"], "mno-sse4.2">, Group<m_x86_Features_Group>;
def mno_sse : Flag<["-"], "mno-sse">, Group<m_x86_Features_Group>;
def mno_ssse3 : Flag<["-"], "mno-ssse3">, Group<m_x86_Features_Group>;
def mno_aes : Flag<["-"], "mno-aes">, Group<m_x86_Features_Group>;
def mno_avx : Flag<["-"], "mno-avx">, Group<m_x86_Features_Group>;
def mno_avx2 : Flag<["-"], "mno-avx2">, Group<m_x86_Features_Group>;
def mno_pclmul : Flag<["-"], "mno-pclmul">, Group<m_x86_Features_Group>;
def mno_lzcnt : Flag<["-"], "mno-lzcnt">, Group<m_x86_Features_Group>;
def mno_rdrnd : Flag<["-"], "mno-rdrnd">, Group<m_x86_Features_Group>;
def mno_bmi : Flag<["-"], "mno-bmi">, Group<m_x86_Features_Group>;
def mno_bmi2 : Flag<["-"], "mno-bmi2">, Group<m_x86_Features_Group>;
def mno_popcnt : Flag<["-"], "mno-popcnt">, Group<m_x86_Features_Group>;
def mno_fma4 : Flag<["-"], "mno-fma4">, Group<m_x86_Features_Group>;
def mno_fma : Flag<["-"], "mno-fma">, Group<m_x86_Features_Group>;
def mno_xop : Flag<["-"], "mno-xop">, Group<m_x86_Features_Group>;
def mno_f16c : Flag<["-"], "mno-f16c">, Group<m_x86_Features_Group>;
def mno_rtm : Flag<["-"], "mno-rtm">, Group<m_x86_Features_Group>;

def mno_thumb : Flag<["-"], "mno-thumb">, Group<m_Group>;
def marm : Flag<["-"], "marm">, Alias<mno_thumb>;

def mno_warn_nonportable_cfstrings : Flag<["-"], "mno-warn-nonportable-cfstrings">, Group<m_Group>;
def mno_omit_leaf_frame_pointer : Flag<["-"], "mno-omit-leaf-frame-pointer">, Group<m_Group>;
def momit_leaf_frame_pointer : Flag<["-"], "momit-leaf-frame-pointer">, Group<m_Group>,
  HelpText<"Omit frame pointer setup for leaf functions">, Flags<[CC1Option]>;
def mpascal_strings : Flag<["-"], "mpascal-strings">, Group<m_Group>;
def mred_zone : Flag<["-"], "mred-zone">, Group<m_Group>;
def mregparm_EQ : Joined<["-"], "mregparm=">, Group<m_Group>;
def mrelax_all : Flag<["-"], "mrelax-all">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"(integrated-as) Relax all machine instructions">;
def mrtd : Flag<["-"], "mrtd">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Make StdCall calling convention the default">;
def msmall_data_threshold_EQ : Joined <["-"], "msmall-data-threshold=">, Group<m_Group>;
def msoft_float : Flag<["-"], "msoft-float">, Group<m_Group>, Flags<[CC1Option]>,
  HelpText<"Use software floating point">;
def mno_implicit_float : Flag<["-"], "mno-implicit-float">, Group<m_Group>,
  HelpText<"Don't generate implicit floating point instructions">;
def msse2 : Flag<["-"], "msse2">, Group<m_x86_Features_Group>;
def msse3 : Flag<["-"], "msse3">, Group<m_x86_Features_Group>;
def msse4a : Flag<["-"], "msse4a">, Group<m_x86_Features_Group>;
def msse4 : Flag<["-"], "msse4">, Group<m_x86_Features_Group>;
def msse4_1 : Flag<["-"], "msse4.1">, Group<m_x86_Features_Group>;
def msse4_2 : Flag<["-"], "msse4.2">, Group<m_x86_Features_Group>;
def msse : Flag<["-"], "msse">, Group<m_x86_Features_Group>;
def mssse3 : Flag<["-"], "mssse3">, Group<m_x86_Features_Group>;
def maes : Flag<["-"], "maes">, Group<m_x86_Features_Group>;
def mavx : Flag<["-"], "mavx">, Group<m_x86_Features_Group>;
def mavx2 : Flag<["-"], "mavx2">, Group<m_x86_Features_Group>;
def mpclmul : Flag<["-"], "mpclmul">, Group<m_x86_Features_Group>;
def mlzcnt : Flag<["-"], "mlzcnt">, Group<m_x86_Features_Group>;
def mrdrnd : Flag<["-"], "mrdrnd">, Group<m_x86_Features_Group>;
def mbmi : Flag<["-"], "mbmi">, Group<m_x86_Features_Group>;
def mbmi2 : Flag<["-"], "mbmi2">, Group<m_x86_Features_Group>;
def mpopcnt : Flag<["-"], "mpopcnt">, Group<m_x86_Features_Group>;
def mfma4 : Flag<["-"], "mfma4">, Group<m_x86_Features_Group>;
def mfma : Flag<["-"], "mfma">, Group<m_x86_Features_Group>;
def mxop : Flag<["-"], "mxop">, Group<m_x86_Features_Group>;
def mf16c : Flag<["-"], "mf16c">, Group<m_x86_Features_Group>;
def mrtm : Flag<["-"], "mrtm">, Group<m_x86_Features_Group>;
def mips16 : Flag<["-"], "mips16">, Group<m_Group>;
def mno_mips16 : Flag<["-"], "mno-mips16">, Group<m_Group>;
def mdsp : Flag<["-"], "mdsp">, Group<m_Group>;
def mno_dsp : Flag<["-"], "mno-dsp">, Group<m_Group>;
def mdspr2 : Flag<["-"], "mdspr2">, Group<m_Group>;
def mno_dspr2 : Flag<["-"], "mno-dspr2">, Group<m_Group>;
def mips32 : Flag<["-"], "mips32">, Group<mips_CPUs_Group>,
  HelpText<"Equivalent to -march=mips32">, Flags<[HelpHidden]>;
def mips32r2 : Flag<["-"], "mips32r2">, Group<mips_CPUs_Group>,
  HelpText<"Equivalent to -march=mips32r2">, Flags<[HelpHidden]>;
def mips64 : Flag<["-"], "mips64">, Group<mips_CPUs_Group>,
  HelpText<"Equivalent to -march=mips64">, Flags<[HelpHidden]>;
def mips64r2 : Flag<["-"], "mips64r2">, Group<mips_CPUs_Group>,
  HelpText<"Equivalent to -march=mips64r2">, Flags<[HelpHidden]>;
def mthumb : Flag<["-"], "mthumb">, Group<m_Group>;
def mtune_EQ : Joined<["-"], "mtune=">, Group<m_Group>;
def multi__module : Flag<["-"], "multi_module">;
def multiply__defined__unused : Separate<["-"], "multiply_defined_unused">;
def multiply__defined : Separate<["-"], "multiply_defined">;
def mwarn_nonportable_cfstrings : Flag<["-"], "mwarn-nonportable-cfstrings">, Group<m_Group>;
def m_Separate : Separate<["-"], "m">, Group<m_Group>;
def m_Joined : Joined<["-"], "m">, Group<m_Group>;
def no_canonical_prefixes : Flag<["-"], "no-canonical-prefixes">, Flags<[HelpHidden]>,
  HelpText<"Use relative instead of canonical paths">;
def no_cpp_precomp : Flag<["-"], "no-cpp-precomp">, Group<clang_ignored_f_Group>;
def no_integrated_as : Flag<["-"], "no-integrated-as">, Flags<[DriverOption]>;
def no_integrated_cpp : Flag<["-", "--"], "no-integrated-cpp">, Flags<[DriverOption]>;
def no_pedantic : Flag<["-", "--"], "no-pedantic">, Group<pedantic_Group>;
def no__dead__strip__inits__and__terms : Flag<["-"], "no_dead_strip_inits_and_terms">;
def nobuiltininc : Flag<["-"], "nobuiltininc">, Flags<[CC1Option]>,
  HelpText<"Disable builtin #include directories">;
def nodefaultlibs : Flag<["-"], "nodefaultlibs">;
def nofixprebinding : Flag<["-"], "nofixprebinding">;
def nolibc : Flag<["-"], "nolibc">;
def nomultidefs : Flag<["-"], "nomultidefs">;
def noprebind : Flag<["-"], "noprebind">;
def noseglinkedit : Flag<["-"], "noseglinkedit">;
def nostartfiles : Flag<["-"], "nostartfiles">;
def nostdinc : Flag<["-"], "nostdinc">;
def nostdlibinc : Flag<["-"], "nostdlibinc">;
def nostdincxx : Flag<["-"], "nostdinc++">, Flags<[CC1Option]>,
  HelpText<"Disable standard #include directories for the C++ standard library">;
def nostdlib : Flag<["-"], "nostdlib">;
def object : Flag<["-"], "object">;
def o : JoinedOrSeparate<["-"], "o">, Flags<[DriverOption, RenderAsInput, CC1Option]>,
  HelpText<"Write output to <file>">, MetaVarName<"<file>">;
def pagezero__size : JoinedOrSeparate<["-"], "pagezero_size">;
def pass_exit_codes : Flag<["-", "--"], "pass-exit-codes">, Flags<[Unsupported]>;
def pedantic_errors : Flag<["-", "--"], "pedantic-errors">, Group<pedantic_Group>, Flags<[CC1Option]>;
def pedantic : Flag<["-", "--"], "pedantic">, Group<pedantic_Group>, Flags<[CC1Option]>;
def pg : Flag<["-"], "pg">, HelpText<"Enable mcount instrumentation">, Flags<[CC1Option]>;
def pipe : Flag<["-", "--"], "pipe">,
  HelpText<"Use pipes between commands, when possible">;
def prebind__all__twolevel__modules : Flag<["-"], "prebind_all_twolevel_modules">;
def prebind : Flag<["-"], "prebind">;
def preload : Flag<["-"], "preload">;
def print_file_name_EQ : Joined<["-", "--"], "print-file-name=">,
  HelpText<"Print the full library path of <file>">, MetaVarName<"<file>">;
def print_ivar_layout : Flag<["-"], "print-ivar-layout">, Flags<[CC1Option]>,
  HelpText<"Enable Objective-C Ivar layout bitmap print trace">;
def print_libgcc_file_name : Flag<["-", "--"], "print-libgcc-file-name">,
  HelpText<"Print the library path for \"libgcc.a\"">;
def print_multi_directory : Flag<["-", "--"], "print-multi-directory">;
def print_multi_lib : Flag<["-", "--"], "print-multi-lib">;
def print_multi_os_directory : Flag<["-", "--"], "print-multi-os-directory">;
def print_prog_name_EQ : Joined<["-", "--"], "print-prog-name=">,
  HelpText<"Print the full program path of <name>">, MetaVarName<"<name>">;
def print_search_dirs : Flag<["-", "--"], "print-search-dirs">,
  HelpText<"Print the paths used for finding libraries and programs">;
def private__bundle : Flag<["-"], "private_bundle">;
def pthreads : Flag<["-"], "pthreads">;
def pthread : Flag<["-"], "pthread">, Flags<[CC1Option]>,
  HelpText<"Support POSIX threads in generated code">;
def p : Flag<["-"], "p">;
def pie : Flag<["-"], "pie">;
def read__only__relocs : Separate<["-"], "read_only_relocs">;
def remap : Flag<["-"], "remap">;
def rewrite_objc : Flag<["-"], "rewrite-objc">, Flags<[DriverOption,CC1Option]>,
  HelpText<"Rewrite Objective-C source to C++">, Group<Action_Group>;
def rewrite_legacy_objc : Flag<["-"], "rewrite-legacy-objc">, Flags<[DriverOption]>,
  HelpText<"Rewrite Legacy Objective-C source to C++">;
def rdynamic : Flag<["-"], "rdynamic">;
def rpath : Separate<["-"], "rpath">, Flags<[LinkerInput]>;
def rtlib_EQ : Joined<["-", "--"], "rtlib=">;
def r : Flag<["-"], "r">;
def save_temps : Flag<["-", "--"], "save-temps">, Flags<[DriverOption]>,
  HelpText<"Save intermediate compilation results">;
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
def shared : Flag<["-", "--"], "shared">;
def single__module : Flag<["-"], "single_module">;
def specs_EQ : Joined<["-", "--"], "specs=">;
def specs : Separate<["-", "--"], "specs">, Flags<[Unsupported]>;
def static_libgcc : Flag<["-"], "static-libgcc">;
def static_libstdcxx : Flag<["-"], "static-libstdc++">;
def static : Flag<["-", "--"], "static">, Flags<[NoArgumentUnused]>;
def std_default_EQ : Joined<["-"], "std-default=">;
def std_EQ : Joined<["-", "--"], "std=">, Flags<[CC1Option]>, Group<L_Group>,
  HelpText<"Language standard to compile for">;
def stdlib_EQ : Joined<["-", "--"], "stdlib=">, Flags<[CC1Option]>,
  HelpText<"C++ standard library to use">;
def sub__library : JoinedOrSeparate<["-"], "sub_library">;
def sub__umbrella : JoinedOrSeparate<["-"], "sub_umbrella">;
def s : Flag<["-"], "s">;
def target : Separate<["-"], "target">, Flags<[DriverOption]>,
  HelpText<"Generate code for the given target">;
def gcc_toolchain : Separate<["-"], "gcc-toolchain">, Flags<[DriverOption]>,
  HelpText<"Use the gcc toolchain at the given directory">;
def time : Flag<["-"], "time">,
  HelpText<"Time individual commands">;
def traditional_cpp : Flag<["-", "--"], "traditional-cpp">, Flags<[CC1Option]>,
  HelpText<"Enable some traditional CPP emulation">;
def traditional : Flag<["-", "--"], "traditional">;
def trigraphs : Flag<["-", "--"], "trigraphs">, Flags<[CC1Option]>,
  HelpText<"Process trigraph sequences">;
def twolevel__namespace__hints : Flag<["-"], "twolevel_namespace_hints">;
def twolevel__namespace : Flag<["-"], "twolevel_namespace">;
def t : Flag<["-"], "t">;
def umbrella : Separate<["-"], "umbrella">;
def undefined : JoinedOrSeparate<["-"], "undefined">, Group<u_Group>;
def undef : Flag<["-"], "undef">, Group<u_Group>, Flags<[CC1Option]>,
  HelpText<"undef all system defines">;
def unexported__symbols__list : Separate<["-"], "unexported_symbols_list">;
def u : JoinedOrSeparate<["-"], "u">, Group<u_Group>;
def use_gold_plugin : Flag<["-"], "use-gold-plugin">;
def v : Flag<["-"], "v">, Flags<[CC1Option]>,
  HelpText<"Show commands to run and use verbose output">;
def verify : Flag<["-"], "verify">, Flags<[DriverOption,CC1Option]>,
  HelpText<"Verify output using a verifier">;
def weak_l : Joined<["-"], "weak-l">, Flags<[LinkerInput]>;
def weak__framework : Separate<["-"], "weak_framework">, Flags<[LinkerInput]>;
def weak__library : Separate<["-"], "weak_library">, Flags<[LinkerInput]>;
def weak__reference__mismatches : Separate<["-"], "weak_reference_mismatches">;
def whatsloaded : Flag<["-"], "whatsloaded">;
def whyload : Flag<["-"], "whyload">;
def w : Flag<["-"], "w">, HelpText<"Suppress all warnings">, Flags<[CC1Option]>;
def x : JoinedOrSeparate<["-"], "x">, Flags<[DriverOption,CC1Option]>,
  HelpText<"Treat subsequent input files as having type <language>">,
  MetaVarName<"<language>">;
def y : Joined<["-"], "y">;

def working_directory : JoinedOrSeparate<["-"], "working-directory">, Flags<[CC1Option]>,
  HelpText<"Resolve file paths relative to the specified directory">;
def working_directory_EQ : Joined<["-"], "working-directory=">, Flags<[CC1Option]>,
  Alias<working_directory>;

// Double dash options, which are usually an alias for one of the previous
// options.

def _CLASSPATH_EQ : Joined<["--"], "CLASSPATH=">, Alias<fclasspath_EQ>;
def _CLASSPATH : Separate<["--"], "CLASSPATH">, Alias<fclasspath_EQ>;
def _all_warnings : Flag<["--"], "all-warnings">, Alias<Wall>;
def _analyze_auto : Flag<["--"], "analyze-auto">, Flags<[DriverOption]>;
def _analyzer_no_default_checks : Flag<["--"], "analyzer-no-default-checks">, Flags<[DriverOption]>;
def _analyzer_output : JoinedOrSeparate<["--"], "analyzer-output">, Flags<[DriverOption]>;
def _analyze : Flag<["--"], "analyze">, Flags<[DriverOption]>,
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
def _help_hidden : Flag<["--"], "help-hidden">;
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
def _machine__EQ : Joined<["--"], "machine-=">, Alias<m_Joined>;
def _machine_ : Joined<["--"], "machine-">, Alias<m_Joined>;
def _machine_EQ : Joined<["--"], "machine=">, Alias<m_Joined>;
def _machine : Separate<["--"], "machine">, Alias<m_Joined>;
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
def _param : Separate<["--"], "param">;
def _param_EQ : Joined<["--"], "param=">, Alias<_param>;
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
def _serialize_diags : Separate<["-", "--"], "serialize-diagnostics">, Flags<[DriverOption]>,
  HelpText<"Serialize compiler diagnostics to a file">;
// We give --version different semantics from -version.
def _version : Flag<["--"], "version">,  Flags<[CC1Option]>;
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
def mieee_rnd_near : Flag<["-"], "mieee-rnd-near">, Group<m_hexagon_Features_Group>;

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

include "CC1Options.td"
