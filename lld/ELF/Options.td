include "llvm/Option/OptParser.td"

// For options whose names are multiple letters, either one dash or
// two can precede the option name except those that start with 'o'.
class F<string name>: Flag<["--", "-"], name>;
class J<string name>: Joined<["--", "-"], name>;
class S<string name>: Separate<["--", "-"], name>;
class JS<string name>: JoinedOrSeparate<["--", "-"], name>;

def Bsymbolic: F<"Bsymbolic">, HelpText<"Bind defined symbols locally">;

def Bsymbolic_functions: F<"Bsymbolic-functions">,
  HelpText<"Bind defined function symbols locally">;

def Bdynamic: F<"Bdynamic">, HelpText<"Link against shared libraries">;

def Bstatic: F<"Bstatic">, HelpText<"Do not link against shared libraries">;

def build_id: F<"build-id">, HelpText<"Generate build ID note">;

def build_id_eq: J<"build-id=">, HelpText<"Generate build ID note">;

def L: JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Add a directory to the library search path">;

def O: Joined<["-"], "O">, HelpText<"Optimize output file size">;

def allow_multiple_definition: F<"allow-multiple-definition">,
  HelpText<"Allow multiple definitions">;

def as_needed: F<"as-needed">,
  HelpText<"Only set DT_NEEDED for shared libraries if used">;

def disable_new_dtags: F<"disable-new-dtags">,
  HelpText<"Disable new dynamic tags">;

def discard_all: F<"discard-all">, HelpText<"Delete all local symbols">;

def discard_locals: F<"discard-locals">,
  HelpText<"Delete temporary local symbols">;

def discard_none: F<"discard-none">,
  HelpText<"Keep all symbols in the symbol table">;

def dynamic_linker: S<"dynamic-linker">,
  HelpText<"Which dynamic linker to use">;

def dynamic_list: S<"dynamic-list">,
  HelpText<"Read a list of dynamic symbols">;

def eh_frame_hdr: F<"eh-frame-hdr">,
  HelpText<"Request creation of .eh_frame_hdr section and PT_GNU_EH_FRAME segment header">;

def enable_new_dtags: F<"enable-new-dtags">,
  HelpText<"Enable new dynamic tags">;

def end_lib: F<"end-lib">,
  HelpText<"End a grouping of objects that should be treated as if they were together in an archive">;

def entry: S<"entry">, MetaVarName<"<entry>">,
  HelpText<"Name of entry point symbol">;

def export_dynamic: F<"export-dynamic">,
  HelpText<"Put symbols in the dynamic symbol table">;

def export_dynamic_symbol: S<"export-dynamic-symbol">,
  HelpText<"Put a symbol in the dynamic symbol table">;

def fatal_warnings: F<"fatal-warnings">,
  HelpText<"Treat warnings as errors">;

def fini: S<"fini">, MetaVarName<"<symbol>">,
  HelpText<"Specify a finalizer function">;

def hash_style: S<"hash-style">,
  HelpText<"Specify hash style (sysv, gnu or both)">;

def help: F<"help">, HelpText<"Print option help">;

def icf: F<"icf=all">, HelpText<"Enable identical code folding">;

def image_base : J<"image-base=">, HelpText<"Set the base address">;

def gc_sections: F<"gc-sections">,
  HelpText<"Enable garbage collection of unused sections">;

def init: S<"init">, MetaVarName<"<symbol>">,
  HelpText<"Specify an initializer function">;

def l: JoinedOrSeparate<["-"], "l">, MetaVarName<"<libName>">,
  HelpText<"Root name of library to use">;

def lto_O: J<"lto-O">, MetaVarName<"<opt-level>">,
  HelpText<"Optimization level for LTO">;

def m: JoinedOrSeparate<["-"], "m">, HelpText<"Set target emulation">;

def no_as_needed: F<"no-as-needed">,
  HelpText<"Always DT_NEEDED for shared libraries">;

def no_demangle: F<"no-demangle">,
  HelpText<"Do not demangle symbol names">;

def no_gnu_unique: F<"no-gnu-unique">,
  HelpText<"Disable STB_GNU_UNIQUE symbol binding">;

def no_whole_archive: F<"no-whole-archive">,
  HelpText<"Restores the default behavior of loading archive members">;

def noinhibit_exec: F<"noinhibit-exec">,
  HelpText<"Retain the executable output file whenever it is still usable">;

def no_undefined: F<"no-undefined">,
  HelpText<"Report unresolved symbols even if the linker is creating a shared library">;

def no_undefined_version: F<"no-undefined-version">,
  HelpText<"Report version scripts that refer undefined symbols">;

def o: JoinedOrSeparate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;

def pie: F<"pie">, HelpText<"Create a position independent executable">;

def print_gc_sections: F<"print-gc-sections">,
  HelpText<"List removed unused sections">;

def reproduce: S<"reproduce">,
  HelpText<"Dump linker invocation and input files for debugging">;

def rpath: S<"rpath">, HelpText<"Add a DT_RUNPATH to the output">;

def relocatable: F<"relocatable">, HelpText<"Create relocatable object file">;

def script: S<"script">, HelpText<"Read linker script">;

def shared: F<"shared">, HelpText<"Build a shared object">;

def soname: J<"soname=">, HelpText<"Set DT_SONAME">;

def start_lib: F<"start-lib">,
  HelpText<"Start a grouping of objects that should be treated as if they were together in an archive">;

def strip_all: F<"strip-all">, HelpText<"Strip all symbols">;

def strip_debug: F<"strip-debug">, HelpText<"Strip debugging information">;

def sysroot: J<"sysroot=">, HelpText<"Set the system root">;

def threads: F<"threads">, HelpText<"Enable use of threads">;

def trace: F<"trace">, HelpText<"Print the names of the input files">;

def trace_symbol : J<"trace-symbol=">, HelpText<"Trace references to symbols">;

def undefined: J<"undefined=">,
  HelpText<"Force undefined symbol during linking">;

def unresolved_symbols: J<"unresolved-symbols=">,
  HelpText<"Determine how to handle unresolved symbols">;

def rsp_quoting: J<"rsp-quoting=">,
  HelpText<"Quoting style for response files. Values supported: windows|posix">;

def verbose: F<"verbose">, HelpText<"Verbose mode">;

def version: F<"version">, HelpText<"Display the version number">;

def version_script: S<"version-script">,
  HelpText<"Read a version script">;

def warn_common: F<"warn-common">,
  HelpText<"Warn about duplicate common symbols">;

def whole_archive: F<"whole-archive">,
  HelpText<"Force load of all members in a static library">;

def wrap: S<"wrap">, MetaVarName<"<symbol>">,
  HelpText<"Use wrapper functions for symbol">;

def z: JoinedOrSeparate<["-"], "z">, MetaVarName<"<option>">,
  HelpText<"Linker option extensions">;

// Aliases
def alias_Bdynamic_call_shared: F<"call_shared">, Alias<Bdynamic>;
def alias_Bdynamic_dy: F<"dy">, Alias<Bdynamic>;
def alias_Bstatic_dn: F<"dn">, Alias<Bstatic>;
def alias_Bstatic_non_shared: F<"non_shared">, Alias<Bstatic>;
def alias_Bstatic_static: F<"static">, Alias<Bstatic>;
def alias_L__library_path: J<"library-path=">, Alias<L>;
def alias_discard_all_x: Flag<["-"], "x">, Alias<discard_all>;
def alias_discard_locals_X: Flag<["-"], "X">, Alias<discard_locals>;
def alias_dynamic_list: J<"dynamic-list=">, Alias<dynamic_list>;
def alias_entry_e: JoinedOrSeparate<["-"], "e">, Alias<entry>;
def alias_entry_entry: J<"entry=">, Alias<entry>;
def alias_export_dynamic_E: Flag<["-"], "E">, Alias<export_dynamic>;
def alias_export_dynamic_symbol: J<"export-dynamic-symbol=">,
  Alias<export_dynamic_symbol>;
def alias_fini_fini: J<"fini=">, Alias<fini>;
def alias_hash_style_hash_style: J<"hash-style=">, Alias<hash_style>;
def alias_init_init: J<"init=">, Alias<init>;
def alias_l__library: J<"library=">, Alias<l>;
def alias_o_output: Joined<["--"], "output=">, Alias<o>;
def alias_pie_pic_executable: F<"pic-executable">, Alias<pie>;
def alias_relocatable_r: Flag<["-"], "r">, Alias<relocatable>;
def alias_rpath_R: Joined<["-"], "R">, Alias<rpath>;
def alias_rpath_rpath: J<"rpath=">, Alias<rpath>;
def alias_script_T: JoinedOrSeparate<["-"], "T">, Alias<script>;
def alias_shared_Bshareable: F<"Bshareable">, Alias<shared>;
def alias_soname_h: JoinedOrSeparate<["-"], "h">, Alias<soname>;
def alias_soname_soname: S<"soname">, Alias<soname>;
def alias_strip_all: Flag<["-"], "s">, Alias<strip_all>;
def alias_strip_debug_S: Flag<["-"], "S">, Alias<strip_debug>;
def alias_trace: Flag<["-"], "t">, Alias<trace>;
def alias_trace_symbol_y : JoinedOrSeparate<["-"], "y">, Alias<trace_symbol>;
def alias_undefined_u: JoinedOrSeparate<["-"], "u">, Alias<undefined>;
def alias_version_V: Flag<["-"], "V">, Alias<version>;
def alias_version_v: Flag<["-"], "v">, Alias<version>;
def alias_wrap_wrap: J<"wrap=">, Alias<wrap>;

// Our symbol resolution algorithm handles symbols in archive files differently
// than traditional linkers, so we don't need --start-group and --end-group.
// These options are recongized for compatibility but ignored.
def end_group: F<"end-group">;
def end_group_paren: Flag<["-"], ")">;
def start_group: F<"start-group">;
def start_group_paren: Flag<["-"], "(">;

// Ignore LTO plugin-related options.
// clang -flto passes -plugin and -plugin-opt to the linker. This is required
// for ld.gold and ld.bfd to get LTO working. But it's not for lld which doesn't
// rely on a plugin. Instead of detecting which linker is used on clang side we
// just ignore the option on lld side as it's easier. In fact, the linker could
// be called 'ld' and understanding which linker is used would require parsing of
// --version output.
def plugin: S<"plugin">;
def plugin_eq: J<"plugin=">;
def plugin_opt: S<"plugin-opt">;
def plugin_opt_eq: J<"plugin-opt=">;

// Options listed below are silently ignored for now for compatibility.
def allow_shlib_undefined: F<"allow-shlib-undefined">;
def define_common: F<"define-common">;
def demangle: F<"demangle">;
def detect_odr_violations: F<"detect-odr-violations">;
def no_add_needed: F<"no-add-needed">;
def no_allow_shlib_undefined: F<"no-allow-shlib-undefined">;
def no_copy_dt_needed_entries: F<"no-copy-dt-needed-entries">,
  Alias<no_add_needed>;
def no_dynamic_linker: F<"no-dynamic-linker">;
def no_fatal_warnings: F<"no-fatal-warnings">;
def no_mmap_output_file: F<"no-mmap-output-file">;
def no_warn_common: F<"no-warn-common">;
def no_warn_mismatch: F<"no-warn-mismatch">;
def rpath_link: S<"rpath-link">;
def rpath_link_eq: J<"rpath-link=">;
def sort_common: F<"sort-common">;
def warn_execstack: F<"warn-execstack">;
def warn_shared_textrel: F<"warn-shared-textrel">;
def G: Separate<["-"], "G">;

// Aliases for ignored options
def alias_define_common_d: Flag<["-"], "d">, Alias<define_common>;
def alias_define_common_dc: F<"dc">, Alias<define_common>;
def alias_define_common_dp: F<"dp">, Alias<define_common>;
def alias_version_script_version_script: J<"version-script=">,
  Alias<version_script>;

// LTO-related options.
def lto_jobs: J<"lto-jobs=">, HelpText<"Number of threads to run codegen">;
def lto_aa_pipeline: J<"lto-aa-pipeline=">,
  HelpText<"AA pipeline to run during LTO. Used in conjunction with -lto-newpm-passes">;
def lto_newpm_passes: J<"lto-newpm-passes=">,
  HelpText<"Passes to run during LTO">;
def disable_verify: F<"disable-verify">;
def mllvm: S<"mllvm">;
def save_temps: F<"save-temps">;
