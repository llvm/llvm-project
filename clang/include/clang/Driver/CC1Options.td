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

let Flags = [CC1Option, NoDriverOption] in {

//===----------------------------------------------------------------------===//
// Target Options
//===----------------------------------------------------------------------===//

def cxx_abi : Separate<["-"], "cxx-abi">,
  HelpText<"Target a particular C++ ABI type">;
def target_abi : Separate<["-"], "target-abi">,
  HelpText<"Target a particular ABI type">;
def target_cpu : Separate<["-"], "target-cpu">,
  HelpText<"Target a specific cpu type">;
def target_feature : Separate<["-"], "target-feature">,
  HelpText<"Target specific attributes">;
def target_linker_version : Separate<["-"], "target-linker-version">,
  HelpText<"Target linker version">;
def triple : Separate<["-"], "triple">,
  HelpText<"Specify target triple (e.g. i686-apple-darwin9)">;
def triple_EQ : Joined<["-"], "triple=">, Alias<triple>;

//===----------------------------------------------------------------------===//
// Analyzer Options
//===----------------------------------------------------------------------===//

def analysis_UnoptimizedCFG : Flag<["-"], "unoptimized-cfg">,
  HelpText<"Generate unoptimized CFGs for all analyses">;
def analysis_CFGAddImplicitDtors : Flag<["-"], "cfg-add-implicit-dtors">,
  HelpText<"Add C++ implicit destructors to CFGs for all analyses">;

def analyzer_store : Separate<["-"], "analyzer-store">,
  HelpText<"Source Code Analysis - Abstract Memory Store Models">;
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
  HelpText<"Force the static analyzer to analyze functions defined in header files">;
def analyzer_opt_analyze_nested_blocks : Flag<["-"], "analyzer-opt-analyze-nested-blocks">,
  HelpText<"Analyze the definitions of blocks in addition to functions">;
def analyzer_display_progress : Flag<["-"], "analyzer-display-progress">,
  HelpText<"Emit verbose output about the analyzer's progress">;
def analyze_function : Separate<["-"], "analyze-function">,
  HelpText<"Run analysis on specific function">;
def analyze_function_EQ : Joined<["-"], "analyze-function=">, Alias<analyze_function>;
def analyzer_eagerly_assume : Flag<["-"], "analyzer-eagerly-assume">,
  HelpText<"Eagerly assume the truth/falseness of some symbolic constraints">;
def trim_egraph : Flag<["-"], "trim-egraph">,
  HelpText<"Only show error-related paths in the analysis graph">;
def analyzer_viz_egraph_graphviz : Flag<["-"], "analyzer-viz-egraph-graphviz">,
  HelpText<"Display exploded graph using GraphViz">;
def analyzer_viz_egraph_ubigraph : Flag<["-"], "analyzer-viz-egraph-ubigraph">,
  HelpText<"Display exploded graph using Ubigraph">;

def analyzer_inline_max_stack_depth : Separate<["-"], "analyzer-inline-max-stack-depth">,
  HelpText<"Bound on stack depth while inlining (4 by default)">;
def analyzer_inline_max_stack_depth_EQ : Joined<["-"], "analyzer-inline-max-stack-depth=">, 
  Alias<analyzer_inline_max_stack_depth>;
  
def analyzer_inline_max_function_size : Separate<["-"], "analyzer-inline-max-function-size">,
  HelpText<"Bound on the number of basic blocks in an inlined function (200 by default)">;
def analyzer_inline_max_function_size_EQ : Joined<["-"], "analyzer-inline-max-function-size=">, 
  Alias<analyzer_inline_max_function_size>;

def analyzer_ipa : Separate<["-"], "analyzer-ipa">,
  HelpText<"Specify the inter-procedural analysis mode">;
def analyzer_ipa_EQ : Joined<["-"], "analyzer-ipa=">, Alias<analyzer_ipa>;
  
def analyzer_inlining_mode : Separate<["-"], "analyzer-inlining-mode">,
  HelpText<"Specify the function selection heuristic used during inlining">;
def analyzer_inlining_mode_EQ : Joined<["-"], "analyzer-inlining-mode=">, Alias<analyzer_inlining_mode>;

def analyzer_disable_retry_exhausted : Flag<["-"], "analyzer-disable-retry-exhausted">,
  HelpText<"Do not re-analyze paths leading to exhausted nodes with a different strategy (may decrease code coverage)">;
  
def analyzer_max_nodes : Separate<["-"], "analyzer-max-nodes">,
  HelpText<"The maximum number of nodes the analyzer can generate (150000 default, 0 = no limit)">;
def analyzer_max_loop : Separate<["-"], "analyzer-max-loop">,
  HelpText<"The maximum number of times the analyzer will go through a loop">;
def analyzer_stats : Flag<["-"], "analyzer-stats">,
  HelpText<"Print internal analyzer statistics.">;

def analyzer_checker : Separate<["-"], "analyzer-checker">,
  HelpText<"Choose analyzer checkers to enable">;
def analyzer_checker_EQ : Joined<["-"], "analyzer-checker=">,
  Alias<analyzer_checker>;

def analyzer_disable_checker : Separate<["-"], "analyzer-disable-checker">,
  HelpText<"Choose analyzer checkers to disable">;
def analyzer_disable_checker_EQ : Joined<["-"], "analyzer-disable-checker=">,
  Alias<analyzer_disable_checker>;

def analyzer_checker_help : Flag<["-"], "analyzer-checker-help">,
  HelpText<"Display the list of analyzer checkers that are available">;

def analyzer_config : Separate<["-"], "analyzer-config">,
  HelpText<"Choose analyzer options to enable">;

//===----------------------------------------------------------------------===//
// Migrator Options
//===----------------------------------------------------------------------===//
def migrator_no_nsalloc_error : Flag<["-"], "no-ns-alloc-error">,
  HelpText<"Do not error on use of NSAllocateCollectable/NSReallocateCollectable">;

def migrator_no_finalize_removal : Flag<["-"], "no-finalize-removal">,
  HelpText<"Do not remove finalize method in gc mode">;

//===----------------------------------------------------------------------===//
// CodeGen Options
//===----------------------------------------------------------------------===//

def disable_llvm_optzns : Flag<["-"], "disable-llvm-optzns">,
  HelpText<"Don't run LLVM optimization passes">;
def disable_llvm_verifier : Flag<["-"], "disable-llvm-verifier">,
  HelpText<"Don't run the LLVM IR verifier pass">;
def disable_red_zone : Flag<["-"], "disable-red-zone">,
  HelpText<"Do not emit code that uses the red zone.">;
def fdebug_compilation_dir : Separate<["-"], "fdebug-compilation-dir">,
  HelpText<"The compilation directory to embed in the debug info.">;
def dwarf_debug_flags : Separate<["-"], "dwarf-debug-flags">,
  HelpText<"The string to embed in the Dwarf debug flags record.">;
def dwarf_column_info : Flag<["-"], "dwarf-column-info">,
  HelpText<"Turn on column location information.">;
def fforbid_guard_variables : Flag<["-"], "fforbid-guard-variables">,
  HelpText<"Emit an error if a C++ static local initializer would need a guard variable">;
def no_implicit_float : Flag<["-"], "no-implicit-float">,
  HelpText<"Don't generate implicit floating point instructions">;
def fdump_vtable_layouts : Flag<["-"], "fdump-vtable-layouts">,
  HelpText<"Dump the layouts of all vtables that will be emitted in a translation unit">;
def femit_coverage_notes : Flag<["-"], "femit-coverage-notes">,
  HelpText<"Emit a gcov coverage notes file when compiling.">;
def femit_coverage_data: Flag<["-"], "femit-coverage-data">,
  HelpText<"Instrument the program to emit gcov coverage data when run.">;
def coverage_file : Separate<["-"], "coverage-file">,
  HelpText<"Emit coverage data to this filename. The extension will be replaced.">;
def coverage_file_EQ : Joined<["-"], "coverage-file=">, Alias<coverage_file>;
def fuse_register_sized_bitfield_access: Flag<["-"], "fuse-register-sized-bitfield-access">,
  HelpText<"Use register sized accesses to bit-fields, when possible.">;
def relaxed_aliasing : Flag<["-"], "relaxed-aliasing">,
  HelpText<"Turn off Type Based Alias Analysis">;
def masm_verbose : Flag<["-"], "masm-verbose">,
  HelpText<"Generate verbose assembly output">;
def mcode_model : Separate<["-"], "mcode-model">,
  HelpText<"The code model to use">;
def mdebug_pass : Separate<["-"], "mdebug-pass">,
  HelpText<"Enable additional debug output">;
def mdisable_fp_elim : Flag<["-"], "mdisable-fp-elim">,
  HelpText<"Disable frame pointer elimination optimization">;
def mdisable_tail_calls : Flag<["-"], "mdisable-tail-calls">,
  HelpText<"Disable tail call optimization, keeping the call stack accurate">;
def menable_no_infinities : Flag<["-"], "menable-no-infs">,
  HelpText<"Allow optimization to assume there are no infinities.">;
def menable_no_nans : Flag<["-"], "menable-no-nans">,
  HelpText<"Allow optimization to assume there are no NaNs.">;
def menable_unsafe_fp_math : Flag<["-"], "menable-unsafe-fp-math">,
  HelpText<"Allow unsafe floating-point math optimizations which may decrease "
           "precision">;
def mfloat_abi : Separate<["-"], "mfloat-abi">,
  HelpText<"The float ABI to use">;
def mlimit_float_precision : Separate<["-"], "mlimit-float-precision">,
  HelpText<"Limit float precision to the given value">;
def mno_exec_stack : Flag<["-"], "mnoexecstack">,
  HelpText<"Mark the file as not needing an executable stack">;
def mno_zero_initialized_in_bss : Flag<["-"], "mno-zero-initialized-in-bss">,
  HelpText<"Do not put zero initialized data in the BSS">;
def backend_option : Separate<["-"], "backend-option">,
  HelpText<"Additional arguments to forward to LLVM backend (during code gen)">;
def mregparm : Separate<["-"], "mregparm">,
  HelpText<"Limit the number of registers available for integer arguments">;
def msave_temp_labels : Flag<["-"], "msave-temp-labels">,
  HelpText<"(integrated-as) Save temporary labels">;
def mrelocation_model : Separate<["-"], "mrelocation-model">,
  HelpText<"The relocation model to use">;
def munwind_tables : Flag<["-"], "munwind-tables">,
  HelpText<"Generate unwinding tables for all functions">;
def fuse_init_array : Flag<["-"], "fuse-init-array">,
  HelpText<"Use .init_array instead of .ctors">;
def mconstructor_aliases : Flag<["-"], "mconstructor-aliases">,
  HelpText<"Emit complete constructors and destructors as aliases when possible">;
def mlink_bitcode_file : Separate<["-"], "mlink-bitcode-file">,
  HelpText<"Link the given bitcode file before performing optimizations.">;

//===----------------------------------------------------------------------===//
// Dependency Output Options
//===----------------------------------------------------------------------===//

def sys_header_deps : Flag<["-"], "sys-header-deps">,
  HelpText<"Include system headers in dependency output">;
def header_include_file : Separate<["-"], "header-include-file">,
  HelpText<"Filename (or -) to write header include output to">;

//===----------------------------------------------------------------------===//
// Diagnostic Options
//===----------------------------------------------------------------------===//

def dump_build_information : Separate<["-"], "dump-build-information">,
  MetaVarName<"<filename>">,
  HelpText<"output a dump of some build information to a file">;
def diagnostic_log_file : Separate<["-"], "diagnostic-log-file">,
  HelpText<"Filename (or -) to log diagnostics to">;
def diagnostic_serialized_file : Separate<["-"], "serialize-diagnostic-file">,
  MetaVarName<"<filename>">,
  HelpText<"File for serializing diagnostics in a binary format">;

def fdiagnostics_format : Separate<["-"], "fdiagnostics-format">,
  HelpText<"Change diagnostic formatting to match IDE and command line tools">;
def fdiagnostics_show_category : Separate<["-"], "fdiagnostics-show-category">,
  HelpText<"Print diagnostic category">;
def ftabstop : Separate<["-"], "ftabstop">, MetaVarName<"<N>">,
  HelpText<"Set the tab stop distance.">;
def ferror_limit : Separate<["-"], "ferror-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of errors to emit before stopping (0 = no limit).">;
def fmacro_backtrace_limit : Separate<["-"], "fmacro-backtrace-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of entries to print in a macro expansion backtrace (0 = no limit).">;
def ftemplate_backtrace_limit : Separate<["-"], "ftemplate-backtrace-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of entries to print in a template instantiation backtrace (0 = no limit).">;
def fconstexpr_backtrace_limit : Separate<["-"], "fconstexpr-backtrace-limit">, MetaVarName<"<N>">,
  HelpText<"Set the maximum number of entries to print in a constexpr evaluation backtrace (0 = no limit).">;
def fmessage_length : Separate<["-"], "fmessage-length">, MetaVarName<"<N>">,
  HelpText<"Format message diagnostics so that they fit within N columns or fewer, when possible.">;
def Wno_rewrite_macros : Flag<["-"], "Wno-rewrite-macros">,
  HelpText<"Silence ObjC rewriting warnings">;

//===----------------------------------------------------------------------===//
// Frontend Options
//===----------------------------------------------------------------------===//

// This isn't normally used, it is just here so we can parse a
// CompilerInvocation out of a driver-derived argument vector.
def cc1 : Flag<["-"], "cc1">;

def ast_merge : Separate<["-"], "ast-merge">,
  MetaVarName<"<ast file>">,
  HelpText<"Merge the given AST file into the translation unit being compiled.">;
def code_completion_at : Separate<["-"], "code-completion-at">,
  MetaVarName<"<file>:<line>:<column>">,
  HelpText<"Dump code-completion information at a location">;
def remap_file : Separate<["-"], "remap-file">,
  MetaVarName<"<from>;<to>">,
  HelpText<"Replace the contents of the <from> file with the contents of the <to> file">;
def code_completion_at_EQ : Joined<["-"], "code-completion-at=">,
  Alias<code_completion_at>;
def code_completion_macros : Flag<["-"], "code-completion-macros">,
  HelpText<"Include macros in code-completion results">;
def code_completion_patterns : Flag<["-"], "code-completion-patterns">,
  HelpText<"Include code patterns in code-completion results">;
def no_code_completion_globals : Flag<["-"], "no-code-completion-globals">,
  HelpText<"Do not include global declarations in code-completion results.">;
def code_completion_brief_comments : Flag<["-"], "code-completion-brief-comments">,
  HelpText<"Include brief documentation comments in code-completion results.">;
def disable_free : Flag<["-"], "disable-free">,
  HelpText<"Disable freeing of memory on exit">;
def load : Separate<["-"], "load">, MetaVarName<"<dsopath>">,
  HelpText<"Load the named plugin (dynamic shared object)">;
def plugin : Separate<["-"], "plugin">, MetaVarName<"<name>">,
  HelpText<"Use the named plugin action instead of the default action (use \"help\" to list available options)">;
def plugin_arg : JoinedAndSeparate<["-"], "plugin-arg-">,
    MetaVarName<"<name> <arg>">,
    HelpText<"Pass <arg> to plugin <name>">;
def add_plugin : Separate<["-"], "add-plugin">, MetaVarName<"<name>">,
  HelpText<"Use the named plugin action in addition to the default action">;
def resource_dir : Separate<["-"], "resource-dir">,
  HelpText<"The directory which holds the compiler resource files">;
def version : Flag<["-"], "version">,
  HelpText<"Print the compiler version">;
def ast_dump_filter : Separate<["-"], "ast-dump-filter">,
  MetaVarName<"<dump_filter>">,
  HelpText<"Use with -ast-dump or -ast-print to dump/print only AST declaration"
           " nodes having a certain substring in a qualified name. Use"
           " -ast-list to list all filterable declaration node names.">;

let Group = Action_Group in {

def Eonly : Flag<["-"], "Eonly">,
  HelpText<"Just run preprocessor, no output (for timings)">;
def dump_raw_tokens : Flag<["-"], "dump-raw-tokens">,
  HelpText<"Lex file in raw mode and dump raw tokens">;
def analyze : Flag<["-"], "analyze">,
  HelpText<"Run static analysis engine">;
def dump_tokens : Flag<["-"], "dump-tokens">,
  HelpText<"Run preprocessor, dump internal rep of tokens">;
def init_only : Flag<["-"], "init-only">,
  HelpText<"Only execute frontend initialization">;
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
def ast_dump_xml : Flag<["-"], "ast-dump-xml">,
  HelpText<"Build ASTs and then debug dump them in a verbose XML format">;
def ast_view : Flag<["-"], "ast-view">,
  HelpText<"Build ASTs and view them with GraphViz">;
def print_decl_contexts : Flag<["-"], "print-decl-contexts">,
  HelpText<"Print DeclContexts and their Decls">;
def emit_module : Flag<["-"], "emit-module">,
  HelpText<"Generate pre-compiled module file from a module map">;
def emit_pth : Flag<["-"], "emit-pth">,
  HelpText<"Generate pre-tokenized header file">;
def emit_pch : Flag<["-"], "emit-pch">,
  HelpText<"Generate pre-compiled header file">;
def emit_llvm_bc : Flag<["-"], "emit-llvm-bc">,
  HelpText<"Build ASTs then convert to LLVM, emit .bc file">;
def emit_llvm_only : Flag<["-"], "emit-llvm-only">,
  HelpText<"Build ASTs and convert to LLVM, discarding output">;
def emit_codegen_only : Flag<["-"], "emit-codegen-only">,
  HelpText<"Generate machine code, but discard output">;
def emit_obj : Flag<["-"], "emit-obj">,
  HelpText<"Emit native object files">;
def rewrite_test : Flag<["-"], "rewrite-test">,
  HelpText<"Rewriter playground">;
def rewrite_macros : Flag<["-"], "rewrite-macros">,
  HelpText<"Expand macros without full preprocessing">;
def migrate : Flag<["-"], "migrate">,
  HelpText<"Migrate source code">;
}

def mt_migrate_directory : Separate<["-"], "mt-migrate-directory">,
  HelpText<"Directory for temporary files produced during ARC or ObjC migration">;
def arcmt_check : Flag<["-"], "arcmt-check">,
  HelpText<"Check for ARC migration issues that need manual handling">;
def arcmt_modify : Flag<["-"], "arcmt-modify">,
  HelpText<"Apply modifications to files to conform to ARC">;
def arcmt_migrate : Flag<["-"], "arcmt-migrate">,
  HelpText<"Apply modifications and produces temporary files that conform to ARC">;

def relocatable_pch : Flag<["-", "--"], "relocatable-pch">,
  HelpText<"Whether to build a relocatable precompiled header">;
def print_stats : Flag<["-"], "print-stats">,
  HelpText<"Print performance metrics and statistics">;
def fdump_record_layouts : Flag<["-"], "fdump-record-layouts">,
  HelpText<"Dump record layout information">;
def fdump_record_layouts_simple : Flag<["-"], "fdump-record-layouts-simple">,
  HelpText<"Dump record layout information in a simple form used for testing">;
def fix_what_you_can : Flag<["-"], "fix-what-you-can">,
  HelpText<"Apply fix-it advice even in the presence of unfixable errors">;
def fix_only_warnings : Flag<["-"], "fix-only-warnings">,
  HelpText<"Apply fix-it advice only for warnings, not errors">;
def fixit_recompile : Flag<["-"], "fixit-recompile">,
  HelpText<"Apply fix-it changes and recompile">;
def fixit_to_temp : Flag<["-"], "fixit-to-temporary">,
  HelpText<"Apply fix-it changes to temporary files">;

def foverride_record_layout_EQ : Joined<["-"], "foverride-record-layout=">,
  HelpText<"Override record layouts with those in the given file">;
  
//===----------------------------------------------------------------------===//
// Language Options
//===----------------------------------------------------------------------===//

def fblocks_runtime_optional : Flag<["-"], "fblocks-runtime-optional">,
  HelpText<"Weakly link in the blocks runtime">;
def fsjlj_exceptions : Flag<["-"], "fsjlj-exceptions">,
  HelpText<"Use SjLj style exceptions">;
def fhidden_weak_vtables : Flag<["-"], "fhidden-weak-vtables">,
  HelpText<"Generate weak vtables and RTTI with hidden visibility">;
def main_file_name : Separate<["-"], "main-file-name">,
  HelpText<"Main file name to use for debug info">;
def fno_signed_char : Flag<["-"], "fno-signed-char">,
  HelpText<"Char is unsigned">;
def fno_wchar : Flag<["-"], "fno-wchar">,
  HelpText<"Disable C++ builtin type wchar_t">;
def fconstant_string_class : Separate<["-"], "fconstant-string-class">,
  MetaVarName<"<class name>">,
  HelpText<"Specify the class to use for constant Objective-C string objects.">;
def fobjc_arc_cxxlib_EQ : Joined<["-"], "fobjc-arc-cxxlib=">,
  HelpText<"Objective-C++ Automatic Reference Counting standard library kind">;
def fobjc_runtime_has_weak : Flag<["-"], "fobjc-runtime-has-weak">,
  HelpText<"The target Objective-C runtime supports ARC weak operations">;
def fobjc_dispatch_method_EQ : Joined<["-"], "fobjc-dispatch-method=">,
  HelpText<"Objective-C dispatch method to use">;
def fobjc_default_synthesize_properties : Flag<["-"], "fobjc-default-synthesize-properties">,
  HelpText<"enable the default synthesis of Objective-C properties">;
def pic_level : Separate<["-"], "pic-level">,
  HelpText<"Value for __PIC__">;
def pie_level : Separate<["-"], "pie-level">,
  HelpText<"Value for __PIE__">;
def fno_validate_pch : Flag<["-"], "fno-validate-pch">,
  HelpText<"Disable validation of precompiled headers">;
def dump_deserialized_pch_decls : Flag<["-"], "dump-deserialized-decls">,
  HelpText<"Dump declarations that are deserialized from PCH, for testing">;
def error_on_deserialized_pch_decl : Separate<["-"], "error-on-deserialized-decl">,
  HelpText<"Emit error if a specific declaration is deserialized from PCH, for testing">;
def error_on_deserialized_pch_decl_EQ : Joined<["-"], "error-on-deserialized-decl=">,
  Alias<error_on_deserialized_pch_decl>;
def static_define : Flag<["-"], "static-define">,
  HelpText<"Should __STATIC__ be defined">;
def stack_protector : Separate<["-"], "stack-protector">,
  HelpText<"Enable stack protectors">;
def stack_protector_buffer_size : Separate<["-"], "stack-protector-buffer-size">,
  HelpText<"Lower bound for a buffer to be considered for stack protection">;
def fvisibility : Separate<["-"], "fvisibility">,
  HelpText<"Default symbol visibility">;
def ftemplate_depth : Separate<["-"], "ftemplate-depth">,
  HelpText<"Maximum depth of recursive template instantiation">;
def fconstexpr_depth : Separate<["-"], "fconstexpr-depth">,
  HelpText<"Maximum depth of recursive constexpr function calls">;
def fconst_strings : Flag<["-"], "fconst-strings">,
  HelpText<"Use a const qualified type for string literals in C and ObjC">;
def fno_const_strings : Flag<["-"], "fno-const-strings">,
  HelpText<"Don't use a const qualified type for string literals in C and ObjC">;
def fno_bitfield_type_align : Flag<["-"], "fno-bitfield-type-align">,
  HelpText<"Ignore bit-field types when aligning structures">;
def ffake_address_space_map : Flag<["-"], "ffake-address-space-map">,
  HelpText<"Use a fake address space map; OpenCL testing purposes only">;
def funknown_anytype : Flag<["-"], "funknown-anytype">,
  HelpText<"Enable parser support for the __unknown_anytype type; for testing purposes only">;
def fdebugger_support : Flag<["-"], "fdebugger-support">,
  HelpText<"Enable special debugger support behavior">;
def fdebugger_cast_result_to_id : Flag<["-"], "fdebugger-cast-result-to-id">,
  HelpText<"Enable casting unknown expression results to id">;
def fdebugger_objc_literal : Flag<["-"], "fdebugger-objc-literal">,
  HelpText<"Enable special debugger support for Objective-C subscripting and literals">;
def fdeprecated_macro : Flag<["-"], "fdeprecated-macro">,
  HelpText<"Defines the __DEPRECATED macro">;
def fno_deprecated_macro : Flag<["-"], "fno-deprecated-macro">,
  HelpText<"Undefines the __DEPRECATED macro">;

//===----------------------------------------------------------------------===//
// Header Search Options
//===----------------------------------------------------------------------===//

def nostdsysteminc : Flag<["-"], "nostdsysteminc">,
  HelpText<"Disable standard system #include directories">;
def fmodule_name : Joined<["-"], "fmodule-name=">, 
  MetaVarName<"<name>">,
  HelpText<"Specify the name of the module to build">;           
def fdisable_module_hash : Flag<["-"], "fdisable-module-hash">,
  HelpText<"Disable the module hash">;
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
def isystem_prefix : JoinedOrSeparate<["-"], "isystem-prefix">,
  MetaVarName<"<prefix>">,
  HelpText<"Treat all #include paths starting with <prefix> as including a "
           "system header.">;
def ino_system_prefix : JoinedOrSeparate<["-"], "ino-system-prefix">,
  MetaVarName<"<prefix>">,
  HelpText<"Treat all #include paths starting with <prefix> as not including a "
           "system header.">;

//===----------------------------------------------------------------------===//
// Preprocessor Options
//===----------------------------------------------------------------------===//

def include_pth : Separate<["-"], "include-pth">, MetaVarName<"<file>">,
  HelpText<"Include file before parsing">;
def chain_include : Separate<["-"], "chain-include">, MetaVarName<"<file>">,
  HelpText<"Include and chain a header file after turning it into PCH">;
def preamble_bytes_EQ : Joined<["-"], "preamble-bytes=">,
  HelpText<"Assume that the precompiled header is a precompiled preamble "
           "covering the first N bytes of the main file">;
def token_cache : Separate<["-"], "token-cache">, MetaVarName<"<path>">,
  HelpText<"Use specified token cache file">;
def detailed_preprocessing_record : Flag<["-"], "detailed-preprocessing-record">,
  HelpText<"include a detailed record of preprocessing actions">;

//===----------------------------------------------------------------------===//
// OpenCL Options
//===----------------------------------------------------------------------===//

def cl_opt_disable : Flag<["-"], "cl-opt-disable">,
  HelpText<"OpenCL only. This option disables all optimizations. The default is optimizations are enabled.">;
def cl_single_precision_constant : Flag<["-"], "cl-single-precision-constant">,
  HelpText<"OpenCL only. Treat double precision floating-point constant as single precision constant.">;
def cl_finite_math_only : Flag<["-"], "cl-finite-math-only">,
  HelpText<"OpenCL only. Allow floating-point optimizations that assume arguments and results are not NaNs or +-Inf.">;
def cl_unsafe_math_optimizations : Flag<["-"], "cl-unsafe-math-optimizations">,
  HelpText<"OpenCL only. Allow unsafe floating-point optimizations.  Also implies -cl-no-signed-zeros and -cl-mad-enable">;
def cl_fast_relaxed_math : Flag<["-"], "cl-fast-relaxed-math">,
  HelpText<"OpenCL only. Sets -cl-finite-math-only and -cl-unsafe-math-optimizations, and defines __FAST_RELAXED_MATH__">;
def cl_mad_enable : Flag<["-"], "cl-mad-enable">,
  HelpText<"OpenCL only. Enable less precise MAD instructions to be generated.">;
def cl_std_EQ : Joined<["-"], "cl-std=">,
  HelpText<"OpenCL language standard to compile for">;

//===----------------------------------------------------------------------===//
// CUDA Options
//===----------------------------------------------------------------------===//

def fcuda_is_device : Flag<["-"], "fcuda-is-device">,
  HelpText<"Generate code for CUDA device">;

} // let Flags = [CC1Option]
