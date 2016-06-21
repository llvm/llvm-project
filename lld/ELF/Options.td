include "llvm/Option/OptParser.td"

// Flag that takes no arguments.
// For options whose names are multiple letters, either one dash or
// two can precede the option name except those that start with 'o'.
class F<string name, string help>: Flag<["--", "-"], name>, HelpText<help>;

def Bsymbolic: F<"Bsymbolic", "Bind defined symbols locally">;
def Bsymbolic_functions: F<"Bsymbolic-functions", "Bind defined function symbols locally">;
def Bdynamic: F<"Bdynamic", "Link against shared libraries">;
def Bstatic: F<"Bstatic", "Do not link against shared libraries">;
def build_id: F<"build-id", "Generate build ID note">;

def build_id_eq : Joined<["--", "-"], "build-id=">;

def L : JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Add a directory to the library search path">;

def O : Joined<["-"], "O">, HelpText<"Optimize">;

def allow_multiple_definition: F<"allow-multiple-definition", "Allow multiple definitions">;
def as_needed: F<"as-needed", "Only set DT_NEEDED for shared libraries if used">;
def disable_new_dtags: F<"disable-new-dtags", "Disable new dynamic tags">;
def discard_all: F<"discard-all", "Delete all local symbols">;
def discard_locals: F<"discard-locals", "Delete temporary local symbols">;
def discard_none: F<"discard-none", "Keep all symbols in the symbol table">;

def dynamic_linker : Separate<["--", "-"], "dynamic-linker">,
  HelpText<"Which dynamic linker to use">;

def dynamic_list : Separate<["--", "-"], "dynamic-list">,
  HelpText<"Read a list of dynamic symbols">;

def eh_frame_hdr: F<"eh-frame-hdr", "Request creation of .eh_frame_hdr section and PT_GNU_EH_FRAME segment header">;
def enable_new_dtags: F<"enable-new-dtags", "Enable new dynamic tags">;
def end_lib: F<"end-lib", "End a grouping of objects that should be treated as if they were together in an archive">;

def entry : Separate<["--", "-"], "entry">, MetaVarName<"<entry>">,
  HelpText<"Name of entry point symbol">;

def export_dynamic: F<"export-dynamic", "Put symbols in the dynamic symbol table">;

def export_dynamic_symbol : Separate<["--", "-"], "export-dynamic-symbol">,
  HelpText<"Put a symbol in the dynamic symbol table">;

def fini : Separate<["-"], "fini">, MetaVarName<"<symbol>">,
  HelpText<"Specify a finalizer function">;

def hash_style : Separate<["--", "-"], "hash-style">,
  HelpText<"Specify hash style (sysv, gnu or both)">;

def help: F<"help", "Print option help">;
def icf: F<"icf=all", "Enable identical code folding">;
def gc_sections: F<"gc-sections", "Enable garbage collection of unused sections">;

def init : Separate<["-"], "init">, MetaVarName<"<symbol>">,
  HelpText<"Specify an initializer function">;

def l : JoinedOrSeparate<["-"], "l">, MetaVarName<"<libName>">,
  HelpText<"Root name of library to use">;

def lto_O : Joined<["--"], "lto-O">, MetaVarName<"<opt-level>">,
  HelpText<"Optimization level for LTO">;

def m : JoinedOrSeparate<["-"], "m">,
  HelpText<"Set target emulation">;

def no_as_needed: F<"no-as-needed", "Always DT_NEEDED for shared libraries">;
def no_demangle: F<"no-demangle", "Do not demangle symbol names">;
def no_gnu_unique: F<"no-gnu-unique", "Disable STB_GNU_UNIQUE symbol binding">;
def no_whole_archive: F<"no-whole-archive", "Restores the default behavior of loading archive members">;
def noinhibit_exec: F<"noinhibit-exec", "Retain the executable output file whenever it is still usable">;
def no_undefined: F<"no-undefined", "Report unresolved symbols even if the linker is creating a shared library">;

def o : JoinedOrSeparate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;

def pie: F<"pie", "Create a position independent executable">;
def print_gc_sections: F<"print-gc-sections", "List removed unused sections">;

def reproduce : Separate<["--"], "reproduce">,
  HelpText<"Dump linker invocation and input files for debugging">;

def rpath : Separate<["-"], "rpath">,
  HelpText<"Add a DT_RUNPATH to the output">;

def relocatable: F<"relocatable", "Create relocatable object file">;

def script : Separate<["--"], "script">,
  HelpText<"Read linker script">;

def shared: F<"shared", "Build a shared object">;

def soname : Joined<["-", "--"], "soname=">,
  HelpText<"Set DT_SONAME">;

def start_lib: F<"start-lib", "Start a grouping of objects that should be treated as if they were together in an archive">;
def strip_all: F<"strip-all", "Strip all symbols">;
def strip_debug: F<"strip-debug", "Strip debugging information">;

def sysroot : Joined<["--"], "sysroot=">,
  HelpText<"Set the system root">;

def threads: F<"threads", "Enable use of threads">;
def trace: F<"trace", "Print the names of the input files">;

def undefined : Joined<["--"], "undefined=">,
  HelpText<"Force undefined symbol during linking">;

def verbose: F<"verbose", "Verbose mode">;
def version: F<"version", "Display the version number">;

def version_script : Separate<["--", "-"], "version-script">,
  HelpText<"Read a version script">;

def warn_common: F<"warn-common", "Warn about duplicate common symbols">;
def whole_archive: F<"whole-archive", "Force load of all members in a static library">;

def wrap : Separate<["--", "-"], "wrap">, MetaVarName<"<symbol>">,
  HelpText<"Use wrapper functions for symbol">;

def z : JoinedOrSeparate<["-"], "z">, MetaVarName<"<option>">,
  HelpText<"Linker option extensions">;

// Aliases
def alias_Bdynamic_call_shared: Flag<["-"], "call_shared">, Alias<Bdynamic>;
def alias_Bdynamic_dy: Flag<["-"], "dy">, Alias<Bdynamic>;
def alias_Bstatic_dn: Flag<["-"], "dn">, Alias<Bstatic>;
def alias_Bstatic_non_shared: Flag<["-"], "non_shared">, Alias<Bstatic>;
def alias_Bstatic_static: Flag<["-"], "static">, Alias<Bstatic>;
def alias_L__library_path : Joined<["--"], "library-path=">, Alias<L>;
def alias_discard_all_x: Flag<["-"], "x">, Alias<discard_all>;
def alias_discard_locals_X: Flag<["-"], "X">, Alias<discard_locals>;
def alias_dynamic_list: Joined<["--", "-"], "dynamic-list=">, Alias<dynamic_list>;
def alias_entry_e : JoinedOrSeparate<["-"], "e">, Alias<entry>;
def alias_export_dynamic_E: Flag<["-"], "E">, Alias<export_dynamic>;
def alias_export_dynamic_symbol : Joined<["--", "-"], "export-dynamic-symbol=">,
    Alias<export_dynamic_symbol>;
def alias_fini_fini : Joined<["-"], "fini=">, Alias<fini>;
def alias_hash_style_hash_style : Joined<["--", "-"], "hash-style=">, Alias<hash_style>;
def alias_init_init : Joined<["-"], "init=">, Alias<init>;
def alias_l__library : Joined<["--"], "library=">, Alias<l>;
def alias_o_output : Joined<["--"], "output=">, Alias<o>;
def alias_pie_pic_executable: Flag<["--"], "pic-executable">, Alias<pie>;
def alias_relocatable_r : Flag<["-"], "r">, Alias<relocatable>;
def alias_rpath_R : Joined<["-"], "R">, Alias<rpath>;
def alias_rpath_rpath : Joined<["-"], "rpath=">, Alias<rpath>;
def alias_script_T : JoinedOrSeparate<["-"], "T">, Alias<script>;
def alias_shared_Bshareable : Flag<["-"], "Bshareable">, Alias<shared>;
def alias_soname_h : JoinedOrSeparate<["-"], "h">, Alias<soname>;
def alias_soname_soname : Separate<["-"], "soname">, Alias<soname>;
def alias_strip_all: Flag<["-"], "s">, Alias<strip_all>;
def alias_strip_debug_S: Flag<["-"], "S">, Alias<strip_debug>;
def alias_trace : Flag<["-"], "t">, Alias<trace>;
def alias_undefined_u : JoinedOrSeparate<["-"], "u">, Alias<undefined>;
def alias_wrap_wrap : Joined<["--", "-"], "wrap=">, Alias<wrap>;

// Our symbol resolution algorithm handles symbols in archive files differently
// than traditional linkers, so we don't need --start-group and --end-group.
// These options are recongized for compatibility but ignored.
def end_group : Flag<["--"], "end-group">;
def end_group_paren: Flag<["-"], ")">;
def start_group : Flag<["--"], "start-group">;
def start_group_paren: Flag<["-"], "(">;

// Ignore LTO plugin-related options.
// clang -flto passes -plugin and -plugin-opt to the linker. This is required
// for ld.gold and ld.bfd to get LTO working. But it's not for lld which doesn't
// rely on a plugin. Instead of detecting which linker is used on clang side we
// just ignore the option on lld side as it's easier. In fact, the linker could
// be called 'ld' and understanding which linker is used would require parsing of
// --version output.
def plugin : Separate<["--", "-"], "plugin">;
def plugin_eq : Joined<["--", "-"], "plugin=">;
def plugin_opt : Separate<["--", "-"], "plugin-opt">;
def plugin_opt_eq : Joined<["--", "-"], "plugin-opt=">;

// Options listed below are silently ignored for now for compatibility.
class Ignore<string name>: Flag<["-", "--"], name>;

def allow_shlib_undefined : Ignore<"allow-shlib-undefined">;
def define_common : Ignore<"define-common">;
def detect_odr_violations : Ignore<"detect-odr-violations">;
def fatal_warnings : Ignore<"fatal-warnings">;
def no_add_needed : Ignore<"no-add-needed">;
def no_allow_shlib_undefined : Ignore<"no-allow-shlib-undefined">;
def no_copy_dt_needed_entries : Ignore<"no-copy-dt-needed-entries">, Alias<no_add_needed>;
def no_dynamic_linker : Ignore<"no-dynamic-linker">;
def no_fatal_warnings : Ignore<"no-fatal-warnings">;
def no_mmap_output_file : Ignore<"no-mmap-output-file">;
def no_undefined_version : Ignore<"no-undefined-version">;
def no_warn_common : Ignore<"no-warn-common">;
def no_warn_mismatch : Ignore<"no-warn-mismatch">;
def rpath_link : Separate<["--", "-"], "rpath-link">;
def rpath_link_eq : Joined<["--", "-"], "rpath-link=">;
def warn_execstack : Ignore<"warn-execstack">;
def warn_shared_textrel : Ignore<"warn-shared-textrel">;
def G : Separate<["-"], "G">;

// Aliases for ignored options
def alias_define_common_d : Ignore<"d">, Alias<define_common>;
def alias_define_common_dc : Ignore<"dc">, Alias<define_common>;
def alias_define_common_dp : Ignore<"dp">, Alias<define_common>;
def alias_version_script_version_script : Joined<["--"], "version-script=">, Alias<version_script>;

// LTO-related options.
def lto_jobs : Joined<["--"], "lto-jobs=">,
  HelpText<"Number of threads to run codegen">;
def lto_aa_pipeline : Joined<["--"], "lto-aa-pipeline=">,
  HelpText<"AA pipeline to run during LTO. Used in conjunction with -lto-newpm-passes">;
def lto_newpm_passes : Joined<["--"], "lto-newpm-passes=">,
  HelpText<"Passes to run during LTO">;
def disable_verify : Flag<["-"], "disable-verify">;
def mllvm : Separate<["-"], "mllvm">;
def save_temps : Flag<["-"], "save-temps">;
