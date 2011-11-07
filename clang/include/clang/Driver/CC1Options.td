//===--- CC1Options.td - Options for clang -cc1 ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by clang -cc1.
//
//===----------------------------------------------------------------------===//

// Include the common option parsing interfaces.
include "OptParser.td"

//===----------------------------------------------------------------------===//
// Target Options
//===----------------------------------------------------------------------===//

def cxx_abi : Separate<"-cxx-abi">,
  HelpText<"Target a particular C++ ABI type">;
def target_abi : Separate<"-target-abi">,
  HelpText<"Target a particular ABI type">;
def target_cpu : Separate<"-target-cpu">,
  HelpText<"Target a specific cpu type">;
def target_feature : Separate<"-target-feature">,
  HelpText<"Target specific attributes">;
def target_linker_version : Separate<"-target-linker-version">,
  HelpText<"Target linker version">;
def triple : Separate<"-triple">,
  HelpText<"Specify target triple (e.g. i686-apple-darwin9)">;
def triple_EQ : Joined<"-triple=">, Alias<triple>;

//===----------------------------------------------------------------------===//
// Analyzer Options
//===----------------------------------------------------------------------===//

def analysis_UnoptimizedCFG : Flag<"-unoptimized-cfg">,
  HelpText<"Generate unoptimized CFGs for all analyses">;
def analysis_CFGAddImplicitDtors : Flag<"-cfg-add-implicit-dtors">,
  HelpText<"Add C++ implicit destructors to CFGs for all analyses">;
def analysis_CFGAddInitializers : Flag<"-cfg-add-initializers">,
  HelpText<"Add C++ initializers to CFGs for all analyses">;

def analyzer_store : Separate<"-analyzer-store">,
  HelpText<"Source Code Analysis - Abstract Memory Store Models">;
def analyzer_store_EQ : Joined<"-analyzer-store=">, Alias<analyzer_store>;

def analyzer_constraints : Separate<"-analyzer-constraints">,
  HelpText<"Source Code Analysis - Symbolic Constraint Engines">;
def analyzer_constraints_EQ : Joined<"-analyzer-constraints=">,
  Alias<analyzer_constraints>;

def analyzer_output : Separate<"-analyzer-output">,
  HelpText<"Source Code Analysis - Output Options">;
def analyzer_output_EQ : Joined<"-analyzer-output=">,
  Alias<analyzer_output>;

def analyzer_purge : Separate<"-analyzer-purge">,
  HelpText<"Source Code Analysis - Dead Symbol Removal Frequency">;
def analyzer_purge_EQ : Joined<"-analyzer-purge=">, Alias<analyzer_purge>;

def analyzer_opt_analyze_headers : Flag<"-analyzer-opt-analyze-headers">,
  HelpText<"Force the static analyzer to analyze functions defined in header files">;
def analyzer_opt_analyze_nested_blocks : Flag<"-analyzer-opt-analyze-nested-blocks">,
  HelpText<"Analyze the definitions of blocks in addition to functions">;
def analyzer_display_progress : Flag<"-analyzer-display-progress">,
  HelpText<"Emit verbose output about the analyzer's progress">;
def analyze_function : Separate<"-analyze-function">,
  HelpText<"Run analysis on specific function">;
def analyze_function_EQ : Joined<"-analyze-function=">, Alias<analyze_function>;
def analyzer_eagerly_assume : Flag<"-analyzer-eagerly-assume">,
  HelpText<"Eagerly assume the truth/falseness of some symbolic constraints">;
def analyzer_no_eagerly_trim_egraph : Flag<"-analyzer-no-eagerly-trim-egraph">,
  HelpText<"Don't eagerly remove uninteresting ExplodedNodes from the ExplodedGraph">;
def trim_egraph : Flag<"-trim-egraph">,
  HelpText<"Only show error-related paths in the analysis graph">;
def analyzer_viz_egraph_graphviz : Flag<"-analyzer-viz-egraph-graphviz">,
  HelpText<"Display exploded graph using GraphViz">;
def analyzer_viz_egraph_ubigraph : Flag<"-analyzer-viz-egraph-ubigraph">,
  HelpText<"Display exploded graph using Ubigraph">;
def analyzer_inline_call : Flag<"-analyzer-inline-call">,
  HelpText<"Experimental transfer function inlining callees when its definition is available.">;
def analyzer_max_nodes : Separate<"-analyzer-max-nodes">,
  HelpText<"The maximum number of nodes the analyzer can generate (150000 default, 0 = no limit)">;
def analyzer_max_loop : Separate<"-analyzer-max-loop">,
  HelpText<"The maximum number of times the analyzer will go through a loop">;

def analyzer_checker : Separate<"-analyzer-checker">,
  HelpText<"Choose analyzer checkers to enable">;
def analyzer_checker_EQ : Joined<"-analyzer-checker=">,
  Alias<analyzer_checker>;

def analyzer_disable_checker : Separate<"-analyzer-disable-checker">,
  HelpText<"Choose analyzer checkers to disable">;
def analyzer_disable_checker_EQ : Joined<"-analyzer-disable-checker=">,
  Alias<analyzer_disable_checker>;

def analyzer_checker_help : Flag<"-analyzer-checker-help">,
  HelpText<"Display the list of analyzer checkers that are available">;

//===----------------------------------------------------------------------===//
// CodeGen Options
//===----------------------------------------------------------------------===//

def disable_llvm_optzns : Flag<"-disable-llvm-optzns">,
  HelpText<"Don't run LLVM optimization passes">;
def disable_llvm_verifier : Flag<"-disable-llvm-verifier">,
  HelpText<"Don't run the LLVM IR verifier pass">;
def disable_red_zone : Flag<"-disable-red-zone">,
  HelpText<"Do not emit code that uses the red zone.">;
def dwarf_debug_flags : Separate<"-dwarf-debug-flags">,
  HelpText<"The string to embed in the Dwarf debug flags record.">;
def fforbid_guard_variables : Flag<"-fforbid-guard-variables">,
  HelpText<"Emit an error if a C++ static local initializer would need a guard variable">;
def g : Flag<"-g">, HelpText<"Generate source level debug information">;
def fno_dwarf2_cfi_asm : Flag<"-fno-dwarf2-cfi-asm">,
  HelpText<"Don't use the cfi directives">;
def fcatch_undefined_behavior : Flag<"-fcatch-undefined-behavior">,
  HelpText<"Generate runtime checks for undefined behavior.">;
def flimit_debug_info : Flag<"-flimit-debug-info">,
  HelpText<"Limit debug information produced to reduce size of debug binary">;
def fno_common : Flag<"-fno-common">,
  HelpText<"Compile common globals like normal definitions">;
def no_implicit_float : Flag<"-no-implicit-float">,
  HelpText<"Don't generate implicit floating point instructions (x86-only)">;
def finstrument_functions : Flag<"-finstrument-functions">,
  HelpText<"Generate calls to instrument function entry and exit">;
def fno_merge_all_constants : Flag<"-fno-merge-all-constants">,
  HelpText<"Disallow merging of constants.">;
def fno_threadsafe_statics : Flag<"-fno-threadsafe-statics">,
  HelpText<"Do not emit code to make initialization of local statics thread safe">;
def fdump_vtable_layouts : Flag<"-fdump-vtable-layouts">,
  HelpText<"Dump the layouts of all vtables that will be emitted in a translation unit">;
def ffunction_sections : Flag<"-ffunction-sections">,
  HelpText<"Place each function in its own section (ELF Only)">;
def fdata_sections : Flag<"-fdata-sections">,
  HelpText<"Place each data in its own section (ELF Only)">;
def funroll_loops : Flag<"-funroll-loops">,
  HelpText<"Turn on loop unroller">;
def femit_coverage_notes : Flag<"-femit-coverage-notes">,
  HelpText<"Emit a gcov coverage notes file when compiling.">;
def femit_coverage_data: Flag<"-femit-coverage-data">,
  HelpText<"Instrument the program to emit gcov coverage data when run.">;
def coverage_file : Separate<"-coverage-file">,
  HelpText<"Emit coverage data to this filename. The extension will be replaced.">;
def coverage_file_EQ : Joined<"-coverage-file=">, Alias<coverage_file>;
def fuse_register_sized_bitfield_access: Flag<"-fuse-register-sized-bitfield-access">,
  HelpText<"Use register sized accesses to bit-fields, when possible.">;
def relaxed_aliasing : Flag<"-relaxed-aliasing">,
  HelpText<"Turn off Type Based Alias Analysis">;
def masm_verbose : Flag<"-masm-verbose">,
  HelpText<"Generate verbose assembly output">;
def mcode_model : Separate<"-mcode-model">,
  HelpText<"The code model to use">;
def mdebug_pass : Separate<"-mdebug-pass">,
  HelpText<"Enable additional debug output">;
def mdisable_fp_elim : Flag<"-mdisable-fp-elim">,
  HelpText<"Disable frame pointer elimination optimization">;
def mfloat_abi : Separate<"-mfloat-abi">,
  HelpText<"The float ABI to use">;
def mno_global_merge : Flag<"-mno-global-merge">,
  HelpText<"Disable merging of globals">;
def mlimit_float_precision : Separate<"-mlimit-float-precision">,
  HelpText<"Limit float precision to the given value">;
def mno_exec_stack : Flag<"-mnoexecstack">,
  HelpText<"Mark the file as not needing an executable stack">;
def mno_zero_initialized_in_bss : Flag<"-mno-zero-initialized-in-bss">,
  HelpText<"Do not put zero initialized data in the BSS">;
def momit_leaf_frame_pointer : Flag<"-momit-leaf-frame-pointer">,
  HelpText<"Omit frame pointer setup for leaf functions.">;
def msoft_float : Flag<"-msoft-float">,
  HelpText<"Use software floating point">;
def backend_option : Separate<"-backend-option">,
  HelpText<"Additional arguments to forward to LLVM backend (during code gen)">;
def mregparm : Separate<"-mregparm">,
  HelpText<"Limit the number of registers available for integer arguments">;
def mrelax_all : Flag<"-mrelax-all">,
  HelpText<"(integrated-as) Relax all machine instructions">;
def msave_temp_labels : Flag<"-msave-temp-labels">,
  HelpText<"(integrated-as) Save temporary labels">;
def mrtd: Flag<"-mrtd">,
  HelpText<"Make StdCall calling convention the default">;
def mrelocation_model : Separate<"-mrelocation-model">,
  HelpText<"The relocation model to use">;
def munwind_tables : Flag<"-munwind-tables">,
  HelpText<"Generate unwinding tables for all functions">;
def mconstructor_aliases : Flag<"-mconstructor-aliases">,
  HelpText<"Emit complete constructors and destructors as aliases when possible">;
def mms_bitfields : Flag<"-mms-bitfields">,
  HelpText<"Set the default structure layout to be compatible with the Microsoft compiler standard.">;
def mstackrealign : Flag<"-mstackrealign">,
  HelpText<"Force realign the stack at entry to every function.">;
def O : Joined<"-O">, HelpText<"Optimization level">;
def Os : Flag<"-Os">, HelpText<"Optimize for size">;
def Oz : Flag<"-Oz">, HelpText<"Optimize for size, regardless of performance">;
def pg : Flag<"-pg">, HelpText<"Enable mcount instrumentation">;

//===----------------------------------------------------------------------===//
// Dependency Output Options
//===----------------------------------------------------------------------===//

def dependency_file : Separate<"-dependency-file">,
  HelpText<"Filename (or -) to write dependency output to">;
def sys_header_deps : Flag<"-sys-header-deps">,
  HelpText<"Include system headers in dependency output">;
def header_include_file : Separate<"-header-include-file">,
  HelpText<"Filename (or -) to write header include output to">;
def H : Flag<"-H">,
  HelpText<"Show header includes and nesting depth">;
def MQ : Separate<"-MQ">, HelpText<"Specify target to quote for dependency">;
def MT : Separate<"-MT">, HelpText<"Specify target for dependency">;
def MP : Flag<"-MP">,
  HelpText<"Create phony target for each dependency (other than main file)">;
def MG : Flag<"-MG">, HelpText<"Add missing headers to dependency list">;

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def dump_build_information : Separate<"-dump-build-information">,
  MetaVarName<"<filename>">,
  HelpText<"output a dump of some build information to a file">;
def diagnostic_log_file : Separate<"-diagnostic-log-file">,
  HelpText<"Filename (or -) to log diagnostics to">;
def fno_show_column : Flag<"-fno-show-column">,
  HelpText<"Do not include column number on diagnostics">;
def fshow_column : Flag<"-fshow-column">,
  HelpText<"Include column number on diagnostics">;
def fno_show_source_location : Flag<"-fno-show-source-location">,
  HelpText<"Do not include source location information with diagnostics">;
def fshow_overloads_EQ : Joined<"-fshow-overloads=">,
  HelpText<"Which overload candidates to show when overload resolution fails: "
           "best|all; defaults to all">;
def fno_caret_diagnostics : Flag<"-fno-caret-diagnostics">,
  HelpText<"Do not include source line and caret with diagnostics">;
def fno_diagnostics_fixit_info : Flag<"-fno-diagnostics-fixit-info">,
  HelpText<"Do not include fixit information in diagnostics">;
def fno_diagnostics_show_note_include_stack :
  Flag<"-fno-diagnostics-show-note-include-stack">,
  HelpText<"Display include stacks for diagnostic notes">;
def w : Flag<"-w">, HelpText<"Suppress all warnings">;
def pedantic : Flag<"-pedantic">;
def pedantic_errors : Flag<"-pedantic-errors">;

// This gets all -W options, including -Werror, -W[no-]system-headers, etc.  The
// driver has stripped off -Wa,foo etc.  The driver has also translated -W to
// -Wextra, so we don't need to worry about it.
def W : Joined<"-W">;

def fdiagnostics_print_source_range_info : Flag<"-fdiagnostics-print-source-range-info">,
  HelpText<"Print source range spans in numeric form">;
def fdiagnostics_parseable_fixits : Flag<"-fdiagnostics-parseable-fixits">,
  HelpText<"Print fix-its in machine parseable form">;
def fdiagnostics_show_name : Flag<"-fdiagnostics-show-name">,
  HelpText<"Print diagnostic name">;
def fdiagnostics_show_option : Flag<"-fdiagnostics-show-option">,
  HelpText<"Print option name with mappable diagnostics">;
def fdiagnostics_format : Separate<"-fdiagnostics-format">,
  HelpText<"Change diagnostic formatting to match IDE and command line tools">;
def fdiagnostics_show_category : Separate<"-fdiagnostics-show-category">,
  HelpText<"Print diagnostic category">;
def fdiagnostics_show_note_include_stack :
  Flag<"-fdiagnostics-show-note-include-stack">,
  HelpText<"Display include stacks for diagnostic notes">;
def ftabstop : Separate<"-ftabstop">, MetaVarName<"<N>">,
  HelpText<"Set the tab stop distance.">;
def ferror_limit : Separate<"-ferror-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of errors to emit before stopping (0 = no limit).">;
def fmacro_backtrace_limit : Separate<"-fmacro-backtrace-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of entries to print in a macro expansion backtrace (0 = no limit).">;
def ftemplate_backtrace_limit : Separate<"-ftemplate-backtrace-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of entries to print in a template instantiation backtrace (0 = no limit).">;
def fmessage_length : Separate<"-fmessage-length">, MetaVarName<"<N>">,
  HelpText<"Format message diagnostics so that they fit within N columns or fewer, when possible.">;
def fcolor_diagnostics : Flag<"-fcolor-diagnostics">,
  HelpText<"Use colors in diagnostics">;
def Wno_rewrite_macros : Flag<"-Wno-rewrite-macros">,
  HelpText<"Silence ObjC rewriting warnings">;
def verify : Flag<"-verify">,
  HelpText<"Verify emitted diagnostics and warnings">;

//===----------------------------------------------------------------------===//
// Frontend Options
//===----------------------------------------------------------------------===//

// This isn't normally used, it is just here so we can parse a
// CompilerInvocation out of a driver-derived argument vector.
def cc1 : Flag<"-cc1">;

def ast_merge : Separate<"-ast-merge">,
  MetaVarName<"<ast file>">,
  HelpText<"Merge the given AST file into the translation unit being compiled.">;
def code_completion_at : Separate<"-code-completion-at">,
  MetaVarName<"<file>:<line>:<column>">,
  HelpText<"Dump code-completion information at a location">;
def remap_file : Separate<"-remap-file">,
  MetaVarName<"<from>;<to>">,
  HelpText<"Replace the contents of the <from> file with the contents of the <to> file">;
def code_completion_at_EQ : Joined<"-code-completion-at=">,
  Alias<code_completion_at>;
def code_completion_macros : Flag<"-code-completion-macros">,
  HelpText<"Include macros in code-completion results">;
def code_completion_patterns : Flag<"-code-completion-patterns">,
  HelpText<"Include code patterns in code-completion results">;
def no_code_completion_globals : Flag<"-no-code-completion-globals">,
  HelpText<"Do not include global declarations in code-completion results.">;
def disable_free : Flag<"-disable-free">,
  HelpText<"Disable freeing of memory on exit">;
def help : Flag<"-help">,
  HelpText<"Print this help text">;
def _help : Flag<"--help">, Alias<help>;
def x : Separate<"-x">, HelpText<"Input language type">;
def o : Separate<"-o">, MetaVarName<"<path>">, HelpText<"Specify output file">;
def load : Separate<"-load">, MetaVarName<"<dsopath>">,
  HelpText<"Load the named plugin (dynamic shared object)">;
def plugin : Separate<"-plugin">, MetaVarName<"<name>">,
  HelpText<"Use the named plugin action instead of the default action (use \"help\" to list available options)">;
def plugin_arg : JoinedAndSeparate<"-plugin-arg-">,
    MetaVarName<"<name> <arg>">,
    HelpText<"Pass <arg> to plugin <name>">;
def add_plugin : Separate<"-add-plugin">, MetaVarName<"<name>">,
  HelpText<"Use the named plugin action in addition to the default action">;
def resource_dir : Separate<"-resource-dir">,
  HelpText<"The directory which holds the compiler resource files">;
def version : Flag<"-version">,
  HelpText<"Print the compiler version">;
def _version : Flag<"--version">, Alias<version>;

def Action_Group : OptionGroup<"<action group>">;
let Group = Action_Group in {

def Eonly : Flag<"-Eonly">,
  HelpText<"Just run preprocessor, no output (for timings)">;
def E : Flag<"-E">,
  HelpText<"Run preprocessor, emit preprocessed file">;
def dump_raw_tokens : Flag<"-dump-raw-tokens">,
  HelpText<"Lex file in raw mode and dump raw tokens">;
def analyze : Flag<"-analyze">,
  HelpText<"Run static analysis engine">;
def dump_tokens : Flag<"-dump-tokens">,
  HelpText<"Run preprocessor, dump internal rep of tokens">;
def init_only : Flag<"-init-only">,
  HelpText<"Only execute frontend initialization">;
def fsyntax_only : Flag<"-fsyntax-only">,
  HelpText<"Run parser and perform semantic analysis">;
def fixit : Flag<"-fixit">,
  HelpText<"Apply fix-it advice to the input source">;
def fixit_EQ : Joined<"-fixit=">,
  HelpText<"Apply fix-it advice creating a file with the given suffix">;
def print_preamble : Flag<"-print-preamble">,
  HelpText<"Print the \"preamble\" of a file, which is a candidate for implicit"
           " precompiled headers.">;
def emit_html : Flag<"-emit-html">,
  HelpText<"Output input source as HTML">;
def ast_print : Flag<"-ast-print">,
  HelpText<"Build ASTs and then pretty-print them">;
def ast_dump : Flag<"-ast-dump">,
  HelpText<"Build ASTs and then debug dump them">;
def ast_dump_xml : Flag<"-ast-dump-xml">,
  HelpText<"Build ASTs and then debug dump them in a verbose XML format">;
def ast_view : Flag<"-ast-view">,
  HelpText<"Build ASTs and view them with GraphViz">;
def print_decl_contexts : Flag<"-print-decl-contexts">,
  HelpText<"Print DeclContexts and their Decls">;
def emit_module : Flag<"-emit-module">,
  HelpText<"Generate pre-compiled module file">;
def emit_pth : Flag<"-emit-pth">,
  HelpText<"Generate pre-tokenized header file">;
def emit_pch : Flag<"-emit-pch">,
  HelpText<"Generate pre-compiled header file">;
def S : Flag<"-S">,
  HelpText<"Emit native assembly code">;
def emit_llvm : Flag<"-emit-llvm">,
  HelpText<"Build ASTs then convert to LLVM, emit .ll file">;
def emit_llvm_bc : Flag<"-emit-llvm-bc">,
  HelpText<"Build ASTs then convert to LLVM, emit .bc file">;
def emit_llvm_only : Flag<"-emit-llvm-only">,
  HelpText<"Build ASTs and convert to LLVM, discarding output">;
def emit_codegen_only : Flag<"-emit-codegen-only">,
  HelpText<"Generate machine code, but discard output">;
def emit_obj : Flag<"-emit-obj">,
  HelpText<"Emit native object files">;
def rewrite_test : Flag<"-rewrite-test">,
  HelpText<"Rewriter playground">;
def rewrite_objc : Flag<"-rewrite-objc">,
  HelpText<"Rewrite ObjC into C (code rewriter example)">;
def rewrite_macros : Flag<"-rewrite-macros">,
  HelpText<"Expand macros without full preprocessing">;
}

def arcmt_check : Flag<"-arcmt-check">,
  HelpText<"Check for ARC migration issues that need manual handling">;
def arcmt_modify : Flag<"-arcmt-modify">,
  HelpText<"Apply modifications to files to conform to ARC">;
def arcmt_migrate : Flag<"-arcmt-migrate">,
  HelpText<"Apply modifications and produces temporary files that conform to ARC">;
def arcmt_migrate_directory : Separate<"-arcmt-migrate-directory">,
  HelpText<"Directory for temporary files produced during ARC migration">;
def arcmt_migrate_report_output : Separate<"-arcmt-migrate-report-output">,
  HelpText<"Output path for the plist report">;
def arcmt_migrate_emit_arc_errors : Flag<"-arcmt-migrate-emit-errors">,
  HelpText<"Emit ARC errors even if the migrator can fix them">;

def working_directory : JoinedOrSeparate<"-working-directory">,
  HelpText<"Resolve file paths relative to the specified directory">;
def working_directory_EQ : Joined<"-working-directory=">,
  Alias<working_directory>;

def relocatable_pch : Flag<"-relocatable-pch">,
  HelpText<"Whether to build a relocatable precompiled header">;
def print_stats : Flag<"-print-stats">,
  HelpText<"Print performance metrics and statistics">;
def ftime_report : Flag<"-ftime-report">,
  HelpText<"Print the amount of time each phase of compilation takes">;
def fdump_record_layouts : Flag<"-fdump-record-layouts">,
  HelpText<"Dump record layout information">;
def fix_what_you_can : Flag<"-fix-what-you-can">,
  HelpText<"Apply fix-it advice even in the presence of unfixable errors">;

// Generic forwarding to LLVM options. This should only be used for debugging
// and experimental features.
def mllvm : Separate<"-mllvm">,
  HelpText<"Additional arguments to forward to LLVM's option processing">;

//===----------------------------------------------------------------------===//
// Language Options
//===----------------------------------------------------------------------===//

def fno_builtin : Flag<"-fno-builtin">,
  HelpText<"Disable implicit builtin knowledge of functions">;
def faltivec : Flag<"-faltivec">,
  HelpText<"Enable AltiVec vector initializer syntax">;
def fno_access_control : Flag<"-fno-access-control">,
  HelpText<"Disable C++ access control">;
def fno_assume_sane_operator_new : Flag<"-fno-assume-sane-operator-new">,
  HelpText<"Don't assume that C++'s global operator new can't alias any pointer">;
def fgnu_keywords : Flag<"-fgnu-keywords">,
  HelpText<"Allow GNU-extension keywords regardless of language standard">;
def fgnu89_inline : Flag<"-fgnu89-inline">,
  HelpText<"Use the gnu89 inline semantics">;
def fno_gnu_keywords : Flag<"-fno-gnu-keywords">,
  HelpText<"Disallow GNU-extension keywords regardless of language standard">;
def fdollars_in_identifiers : Flag<"-fdollars-in-identifiers">,
  HelpText<"Allow '$' in identifiers">;
def fno_dollars_in_identifiers : Flag<"-fno-dollars-in-identifiers">,
  HelpText<"Disallow '$' in identifiers">;
def femit_all_decls : Flag<"-femit-all-decls">,
  HelpText<"Emit all declarations, even if unused">;
def fblocks : Flag<"-fblocks">,
  HelpText<"Enable the 'blocks' language feature">;
def fblocks_runtime_optional : Flag<"-fblocks-runtime-optional">,
  HelpText<"Weakly link in the blocks runtime">;
def fheinous_gnu_extensions : Flag<"-fheinous-gnu-extensions">;
def fexceptions : Flag<"-fexceptions">,
  HelpText<"Enable support for exception handling">;
def fobjc_exceptions : Flag<"-fobjc-exceptions">,
  HelpText<"Enable Objective-C exceptions">;
def fcxx_exceptions : Flag<"-fcxx-exceptions">,
  HelpText<"Enable C++ exceptions">;
def fsjlj_exceptions : Flag<"-fsjlj-exceptions">,
  HelpText<"Use SjLj style exceptions">;
def ffreestanding : Flag<"-ffreestanding">,
  HelpText<"Assert that the compilation takes place in a freestanding environment">;
def fgnu_runtime : Flag<"-fgnu-runtime">,
  HelpText<"Generate output compatible with the standard GNU Objective-C runtime">;
def fhidden_weak_vtables : Flag<"-fhidden-weak-vtables">,
  HelpText<"Generate weak vtables and RTTI with hidden visibility">;
def std_EQ : Joined<"-std=">,
  HelpText<"Language standard to compile for">;
def stdlib_EQ : Joined<"-stdlib=">,
  HelpText<"C++ standard library to use">;
def fmath_errno : Flag<"-fmath-errno">,
  HelpText<"Require math functions to indicate errors by setting errno">;
def fms_extensions : Flag<"-fms-extensions">,
  HelpText<"Accept some non-standard constructs supported by the Microsoft compiler">;
def fms_compatibility : Flag<"-fms-compatibility">,
  HelpText<"Enable Microsoft compatibility mode">;
def fmsc_version : Joined<"-fmsc-version=">,
  HelpText<"Version of the Microsoft C/C++ compiler to report in _MSC_VER (0 = don't define it (default))">;
def fborland_extensions : Flag<"-fborland-extensions">,
  HelpText<"Accept non-standard constructs supported by the Borland compiler">;
def main_file_name : Separate<"-main-file-name">,
  HelpText<"Main file name to use for debug info">;
def fno_elide_constructors : Flag<"-fno-elide-constructors">,
  HelpText<"Disable C++ copy constructor elision">;
def fno_lax_vector_conversions : Flag<"-fno-lax-vector-conversions">,
  HelpText<"Disallow implicit conversions between vectors with a different number of elements or different element types">;
def fno_operator_names : Flag<"-fno-operator-names">,
  HelpText<"Do not treat C++ operator name keywords as synonyms for operators">;
def fno_signed_char : Flag<"-fno-signed-char">,
  HelpText<"Char is unsigned">;
def fno_spell_checking : Flag<"-fno-spell-checking">,
  HelpText<"Disable spell-checking">;
def fno_use_cxa_atexit : Flag<"-fno-use-cxa-atexit">,
  HelpText<"Don't use __cxa_atexit for calling destructors">;
def fconstant_string_class : Separate<"-fconstant-string-class">,
  MetaVarName<"<class name>">,
  HelpText<"Specify the class to use for constant Objective-C string objects.">;
def fno_constant_cfstrings : Flag<"-fno-constant-cfstrings">,
  HelpText<"Enable creation of CodeFoundation-type constant strings">;
def fobjc_arc : Flag<"-fobjc-arc">,
  HelpText<"Synthesize retain and release calls for Objective-C pointers">;
def fobjc_arc_cxxlib_EQ : Joined<"-fobjc-arc-cxxlib=">,
  HelpText<"Objective-C++ Automatic Reference Counting standard library kind">;
def fobjc_arc_exceptions : Flag<"-fobjc-arc-exceptions">,
  HelpText<"Use EH-safe code when synthesizing retains and releases in -fobjc-arc">;
def fobjc_runtime_has_arc : Flag<"-fobjc-runtime-has-arc">,
  HelpText<"The target Objective-C runtime provides ARC entrypoints">;
def fobjc_runtime_has_weak : Flag<"-fobjc-runtime-has-weak">,
  HelpText<"The target Objective-C runtime supports ARC weak operations">;
def fobjc_runtime_has_terminate : Flag<"-fobjc-runtime-has-terminate">,
  HelpText<"The target Objective-C runtime provides an objc_terminate entrypoint">;
def fobjc_gc : Flag<"-fobjc-gc">,
  HelpText<"Enable Objective-C garbage collection">;
def fobjc_gc_only : Flag<"-fobjc-gc-only">,
  HelpText<"Use GC exclusively for Objective-C related memory management">;
def fapple_kext : Flag<"-fapple-kext">,
  HelpText<"Use Apple's kernel extensions ABI">;
def fobjc_dispatch_method_EQ : Joined<"-fobjc-dispatch-method=">,
  HelpText<"Objective-C dispatch method to use">;
def fobjc_default_synthesize_properties : Flag<"-fobjc-default-synthesize-properties">,
  HelpText<"enable the default synthesis of Objective-C properties">;
def print_ivar_layout : Flag<"-print-ivar-layout">,
  HelpText<"Enable Objective-C Ivar layout bitmap print trace">;
def fobjc_fragile_abi : Flag<"-fobjc-fragile-abi">,
  HelpText<"Use Objective-C's fragile ABI">;
def fno_objc_infer_related_result_type : Flag<
    "-fno-objc-infer-related-result-type">,
  HelpText<
    "do not infer Objective-C related result type based on method family">;
def ftrapv : Flag<"-ftrapv">,
  HelpText<"Trap on integer overflow">;
def ftrapv_handler : Separate<"-ftrapv-handler">,
  MetaVarName<"<function name>">,
  HelpText<"Specify the function to be called on overflow.">;
def fwrapv : Flag<"-fwrapv">,
  HelpText<"Treat signed integer overflow as two's complement">;
def pic_level : Separate<"-pic-level">,
  HelpText<"Value for __PIC__">;
def pthread : Flag<"-pthread">,
  HelpText<"Support POSIX threads in generated code">;
def fpack_struct : Separate<"-fpack-struct">,
  HelpText<"Specify the default maximum struct packing alignment">;
def fpascal_strings : Flag<"-fpascal-strings">,
  HelpText<"Recognize and construct Pascal-style string literals">;
def fno_rtti : Flag<"-fno-rtti">,
  HelpText<"Disable generation of rtti information">;
def fno_validate_pch : Flag<"-fno-validate-pch">,
  HelpText<"Disable validation of precompiled headers">;
def dump_deserialized_pch_decls : Flag<"-dump-deserialized-decls">,
  HelpText<"Dump declarations that are deserialized from PCH, for testing">;
def error_on_deserialized_pch_decl : Separate<"-error-on-deserialized-decl">,
  HelpText<"Emit error if a specific declaration is deserialized from PCH, for testing">;
def fshort_wchar : Flag<"-fshort-wchar">,
  HelpText<"Force wchar_t to be a short unsigned int">;
def fshort_enums : Flag<"-fshort-enums">,
  HelpText<"Allocate to an enum type only as many bytes as it needs for the declared range of possible values">;
def static_define : Flag<"-static-define">,
  HelpText<"Should __STATIC__ be defined">;
def stack_protector : Separate<"-stack-protector">,
  HelpText<"Enable stack protectors">;
def fvisibility : Separate<"-fvisibility">,
  HelpText<"Default symbol visibility">;
def fvisibility_inlines_hidden : Flag<"-fvisibility-inlines-hidden">,
  HelpText<"Give inline C++ member functions default visibility by default">;
def ftemplate_depth : Separate<"-ftemplate-depth">,
  HelpText<"Maximum depth of recursive template instantiation">;
def Wlarge_by_value_copy : Separate<"-Wlarge-by-value-copy">,
  HelpText<"Warn if a function definition returns or accepts an object larger "
           "in bytes that a given value">;
def Wlarge_by_value_copy_EQ : Joined<"-Wlarge-by-value-copy=">,
  Alias<Wlarge_by_value_copy>;
def trigraphs : Flag<"-trigraphs">,
  HelpText<"Process trigraph sequences">;
def fwritable_strings : Flag<"-fwritable-strings">,
  HelpText<"Store string literals as writable data">;
def fconst_strings : Flag<"-fconst-strings">,
  HelpText<"Use a const qualified type for string literals in C and ObjC">;
def fno_const_strings : Flag<"-fno-const-strings">,
  HelpText<"Don't use a const qualified type for string literals in C and ObjC">;
def fno_bitfield_type_align : Flag<"-fno-bitfield-type-align">,
  HelpText<"Ignore bit-field types when aligning structures">;
def traditional_cpp : Flag<"-traditional-cpp">,
  HelpText<"Enable some traditional CPP emulation">;
def ffake_address_space_map : Flag<"-ffake-address-space-map">,
  HelpText<"Use a fake address space map; OpenCL testing purposes only">;
def fdelayed_template_parsing : Flag<"-fdelayed-template-parsing">,
  HelpText<"Parse templated function definitions at the end of the "
           "translation unit ">;
def funknown_anytype : Flag<"-funknown-anytype">,
  HelpText<"Enable parser support for the __unknown_anytype type; for testing purposes only">;
def fdebugger_support : Flag<"-fdebugger-support">,
  HelpText<"Enable special debugger support behavior">;
def fdeprecated_macro : Flag<"-fdeprecated-macro">,
  HelpText<"Defines the __DEPRECATED macro">;
def fno_deprecated_macro : Flag<"-fno-deprecated-macro">,
  HelpText<"Undefines the __DEPRECATED macro">;

//===----------------------------------------------------------------------===//
// Header Search Options
//===----------------------------------------------------------------------===//

def nostdsysteminc : Flag<"-nostdsysteminc">,
  HelpText<"Disable standard system #include directories">;
def nostdincxx : Flag<"-nostdinc++">,
  HelpText<"Disable standard #include directories for the C++ standard library">;
def nobuiltininc : Flag<"-nobuiltininc">,
  HelpText<"Disable builtin #include directories">;
def fmodule_cache_path : Separate<"-fmodule-cache-path">, 
  MetaVarName<"<directory>">,
  HelpText<"Specify the module cache path">;           
def fdisable_module_hash : Flag<"-fdisable-module-hash">,
  HelpText<"Disable the module hash">;
def fauto_module_import : Flag<"-fauto-module-import">,
  HelpText<"Automatically translate #include/#import into module imports "
           "when possible">;
           
def F : JoinedOrSeparate<"-F">, MetaVarName<"<directory>">,
  HelpText<"Add directory to framework include search path">;
def I : JoinedOrSeparate<"-I">, MetaVarName<"<directory>">,
  HelpText<"Add directory to include search path">;
def idirafter : JoinedOrSeparate<"-idirafter">, MetaVarName<"<directory>">,
  HelpText<"Add directory to AFTER include search path">;
def index_header_map : Flag<"-index-header-map">, 
  HelpText<"Make the next included directory (-I or -F) an indexer header map">;
def iquote : JoinedOrSeparate<"-iquote">, MetaVarName<"<directory>">,
  HelpText<"Add directory to QUOTE include search path">;
def c_isystem : JoinedOrSeparate<"-c-isystem">, MetaVarName<"<directory>">,
  HelpText<"Add directory to the C SYSTEM include search path">;
def cxx_isystem : JoinedOrSeparate<"-cxx-isystem">, MetaVarName<"<directory>">,
  HelpText<"Add directory to the C++ SYSTEM include search path">;
def objc_isystem : JoinedOrSeparate<"-objc-isystem">,
  MetaVarName<"<directory>">,
  HelpText<"Add directory to the ObjC SYSTEM include search path">;
def objcxx_isystem : JoinedOrSeparate<"-objcxx-isystem">,
  MetaVarName<"<directory>">,
  HelpText<"Add directory to the ObjC++ SYSTEM include search path">;
def isystem : JoinedOrSeparate<"-isystem">, MetaVarName<"<directory>">,
  HelpText<"Add directory to SYSTEM include search path">;
def iwithsysroot : JoinedOrSeparate<"-iwithsysroot">,MetaVarName<"<directory>">,
  HelpText<"Add directory to SYSTEM include search path, "
           "absolute paths are relative to -isysroot">;
def internal_isystem : JoinedOrSeparate<"-internal-isystem">,
  MetaVarName<"<directory>">,
  HelpText<"Add directory to the internal system include search path; these "
           "are assumed to not be user-provided and are used to model system "
           "and standard headers' paths.">;
def internal_externc_isystem : JoinedOrSeparate<"-internal-externc-isystem">,
  MetaVarName<"<directory>">,
  HelpText<"Add directory to the internal system include search path with "
           "implicit extern \"C\" semantics; these are assumed to not be "
           "user-provided and are used to model system and standard headers' "
           "paths.">;
def iprefix : JoinedOrSeparate<"-iprefix">, MetaVarName<"<prefix>">,
  HelpText<"Set the -iwithprefix/-iwithprefixbefore prefix">;
def iwithprefix : JoinedOrSeparate<"-iwithprefix">, MetaVarName<"<dir>">,
  HelpText<"Set directory to SYSTEM include search path with prefix">;
def iwithprefixbefore : JoinedOrSeparate<"-iwithprefixbefore">,
  MetaVarName<"<dir>">,
  HelpText<"Set directory to include search path with prefix">;
def isysroot : JoinedOrSeparate<"-isysroot">, MetaVarName<"<dir>">,
  HelpText<"Set the system root directory (usually /)">;
def v : Flag<"-v">, HelpText<"Enable verbose output">;

//===----------------------------------------------------------------------===//
// Preprocessor Options
//===----------------------------------------------------------------------===//

def D : JoinedOrSeparate<"-D">, MetaVarName<"<macro>">,
  HelpText<"Predefine the specified macro">;
def include_ : JoinedOrSeparate<"-include">, MetaVarName<"<file>">, EnumName<"include">,
  HelpText<"Include file before parsing">;
def imacros : JoinedOrSeparate<"-imacros">, MetaVarName<"<file>">,
  HelpText<"Include macros from file before parsing">;
def include_pch : Separate<"-include-pch">, MetaVarName<"<file>">,
  HelpText<"Include precompiled header file">;
def include_pth : Separate<"-include-pth">, MetaVarName<"<file>">,
  HelpText<"Include file before parsing">;
def chain_include : Separate<"-chain-include">, MetaVarName<"<file>">,
  HelpText<"Include and chain a header file after turning it into PCH">;
def preamble_bytes_EQ : Joined<"-preamble-bytes=">,
  HelpText<"Assume that the precompiled header is a precompiled preamble "
           "covering the first N bytes of the main file">;
def token_cache : Separate<"-token-cache">, MetaVarName<"<path>">,
  HelpText<"Use specified token cache file">;
def U : JoinedOrSeparate<"-U">, MetaVarName<"<macro>">,
  HelpText<"Undefine the specified macro">;
def undef : Flag<"-undef">, MetaVarName<"<macro>">,
  HelpText<"undef all system defines">;
def detailed_preprocessing_record : Flag<"-detailed-preprocessing-record">,
  HelpText<"include a detailed record of preprocessing actions">;

//===----------------------------------------------------------------------===//
// Preprocessed Output Options
//===----------------------------------------------------------------------===//

def P : Flag<"-P">,
  HelpText<"Disable linemarker output in -E mode">;
def C : Flag<"-C">,
  HelpText<"Enable comment output in -E mode">;
def CC : Flag<"-CC">,
  HelpText<"Enable comment output in -E mode, even from macro expansions">;
def dM : Flag<"-dM">,
  HelpText<"Print macro definitions in -E mode instead of normal output">;
def dD : Flag<"-dD">,
  HelpText<"Print macro definitions in -E mode in addition to normal output">;

//===----------------------------------------------------------------------===//
// OpenCL Options
//===----------------------------------------------------------------------===//

def cl_opt_disable : Flag<"-cl-opt-disable">,
  HelpText<"OpenCL only. This option disables all optimizations. The default is optimizations are enabled.">;
def cl_single_precision_constant : Flag<"-cl-single-precision-constant">,
  HelpText<"OpenCL only. Treat double precision floating-point constant as single precision constant.">;
def cl_finite_math_only : Flag<"-cl-finite-math-only">,
  HelpText<"OpenCL only. Allow floating-point optimizations that assume arguments and results are not NaNs or +-Inf.">;
def cl_unsafe_math_optimizations : Flag<"-cl-unsafe-math-optimizations">,
  HelpText<"OpenCL only. Allow unsafe floating-point optimizations.  Also implies -cl-no-signed-zeros and -cl-mad-enable">;
def cl_fast_relaxed_math : Flag<"-cl-fast-relaxed-math">,
  HelpText<"OpenCL only. Sets -cl-finite-math-only and -cl-unsafe-math-optimizations, and defines __FAST_RELAXED_MATH__">;
def cl_mad_enable : Flag<"-cl-mad-enable">,
  HelpText<"OpenCL only. Enable less precise MAD instructions to be generated.">;
def cl_std_EQ : Joined<"-cl-std=">,
  HelpText<"OpenCL language standard to compile for">;

//===----------------------------------------------------------------------===//
// CUDA Options
//===----------------------------------------------------------------------===//

def fcuda_is_device : Flag<"-fcuda-is-device">,
  HelpText<"Generate code for CUDA device">;
