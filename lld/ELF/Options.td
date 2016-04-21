include "llvm/Option/OptParser.td"

def Bsymbolic: Flag<["-"], "Bsymbolic">,
  HelpText<"Bind defined symbols locally">;

def Bsymbolic_functions: Flag<["-"], "Bsymbolic-functions">,
  HelpText<"Bind defined function symbols locally">;

def Bdynamic: Flag<["-"], "Bdynamic">,
  HelpText<"Link against shared libraries">;

def Bstatic: Flag<["-"], "Bstatic">,
  HelpText<"Do not link against shared libraries">;

def build_id : Flag<["--", "-"], "build-id">,
  HelpText<"Generate build ID note">;

def build_id_eq : Joined<["--", "-"], "build-id=">;

def L : JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Directory to search for libraries">;

def O : Joined<["-"], "O">, HelpText<"Optimize">;

def allow_multiple_definition: Flag<["--"], "allow-multiple-definition">,
  HelpText<"Allow multiple definitions">;

def allow_shlib_undefined : Flag<["--", "-"], "allow-shlib-undefined">;

def as_needed : Flag<["--"], "as-needed">;

def disable_new_dtags : Flag<["--"], "disable-new-dtags">,
  HelpText<"Disable new dynamic tags">;

def discard_all : Flag<["-"], "discard-all">,
  HelpText<"Delete all local symbols">;

def discard_locals : Flag<["-"], "discard-locals">,
  HelpText<"Delete temporary local symbols">;

def discard_none : Flag<["-"], "discard-none">,
  HelpText<"Keep all symbols in the symbol table">;

def dynamic_linker : Separate<["--", "-"], "dynamic-linker">,
  HelpText<"Which dynamic linker to use">;

def dynamic_list : Separate<["--", "-"], "dynamic-list">,
  HelpText<"Read a list of dynamic symbols">;

def eh_frame_hdr : Flag<["--"], "eh-frame-hdr">,
  HelpText<"Request creation of .eh_frame_hdr section and PT_GNU_EH_FRAME segment header">;

def enable_new_dtags : Flag<["--"], "enable-new-dtags">,
  HelpText<"Enable new dynamic tags">;

def end_lib : Flag<["--"], "end-lib">,
  HelpText<"End a library">;

def entry : Separate<["--", "-"], "entry">, MetaVarName<"<entry>">,
  HelpText<"Name of entry point symbol">;

def export_dynamic : Flag<["--", "-"], "export-dynamic">,
  HelpText<"Put symbols in the dynamic symbol table">;

def fini : Separate<["-"], "fini">, MetaVarName<"<symbol>">,
  HelpText<"Specify a finalizer function">;

def hash_style : Separate<["--", "-"], "hash-style">,
  HelpText<"Specify hash style (sysv, gnu or both)">;

def help : Flag<["--", "-"], "help">,
  HelpText<"Print option help">;

def icf : Flag<["--"], "icf=all">,
  HelpText<"Enable identical code folding">;

def gc_sections : Flag<["--"], "gc-sections">,
  HelpText<"Enable garbage collection of unused sections">;

def init : Separate<["-"], "init">, MetaVarName<"<symbol>">,
  HelpText<"Specify an initializer function">;

def l : JoinedOrSeparate<["-"], "l">, MetaVarName<"<libName>">,
  HelpText<"Root name of library to use">;

def lto_O : Joined<["--"], "lto-O">, MetaVarName<"<opt-level>">,
  HelpText<"Optimization level for LTO">;

def m : JoinedOrSeparate<["-"], "m">,
  HelpText<"Set target emulation">;

def no_allow_shlib_undefined : Flag<["--"], "no-allow-shlib-undefined">;

def no_as_needed : Flag<["--"], "no-as-needed">;

def no_demangle: Flag<["--"], "no-demangle">,
  HelpText<"Do not demangle symbol names">;

def no_gnu_unique : Flag<["--"], "no-gnu-unique">,
  HelpText<"Disable STB_GNU_UNIQUE symbol binding">;

def no_whole_archive : Flag<["--", "-"], "no-whole-archive">,
  HelpText<"Restores the default behavior of loading archive members">;

def noinhibit_exec : Flag<["--"], "noinhibit-exec">,
  HelpText<"Retain the executable output file whenever it is still usable">;

def no_undefined : Flag<["--"], "no-undefined">,
  HelpText<"Report unresolved symbols even if the linker is creating a shared library">;

def o : JoinedOrSeparate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;

def pie : Flag<["-"], "pie">,
  HelpText<"Create a position independent executable">;

def print_gc_sections: Flag<["--"], "print-gc-sections">,
  HelpText<"List removed unused sections">;

def rpath : Separate<["-"], "rpath">,
  HelpText<"Add a DT_RUNPATH to the output">;

def relocatable : Flag<["--"], "relocatable">,
  HelpText<"Create relocatable object file">;

def script : Separate<["--"], "script">,
  HelpText<"Read linker script">;

def shared : Flag<["-"], "shared">,
  HelpText<"Build a shared object">;

def soname : Joined<["-"], "soname=">,
  HelpText<"Set DT_SONAME">;

def start_lib : Flag<["--"], "start-lib">,
  HelpText<"Start a library">;

def strip_all : Flag<["--"], "strip-all">,
  HelpText<"Strip all symbols">;

def strip_debug : Flag<["--"], "strip-debug">,
  HelpText<"Strip debugging information">;

def sysroot : Joined<["--"], "sysroot=">,
  HelpText<"Set the system root">;

def threads : Joined<["--"], "threads">;

def trace: Flag<["--"], "trace">,
  HelpText<"Print the names of the input files">;

def undefined : Joined<["--"], "undefined=">,
  HelpText<"Force undefined symbol during linking">;

def verbose : Flag<["--"], "verbose">;

def version : Flag<["--", "-"], "version">,
  HelpText<"Display the version number">;

def warn_common : Flag<["--", "-"], "warn-common">,
  HelpText<"Warn about duplicate common symbols">;

def whole_archive : Flag<["--", "-"], "whole-archive">,
  HelpText<"Force load of all members in a static library">;

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
def alias_fini_fini : Joined<["-"], "fini=">, Alias<fini>;
def alias_hash_style_hash_style : Joined<["--", "-"], "hash-style=">, Alias<hash_style>;
def alias_init_init : Joined<["-"], "init=">, Alias<init>;
def alias_l__library : Joined<["--"], "library=">, Alias<l>;
def alias_o_output : Joined<["--"], "output=">, Alias<o>;
def alias_pie_pic_executable: Flag<["--"], "pic-executable">, Alias<pie>;
def alias_relocatable_r : Flag<["-"], "r">, Alias<relocatable>;
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
def detect_odr_violations : Flag<["--"], "detect-odr-violations">;
def fatal_warnings : Flag<["--"], "fatal-warnings">;
def no_add_needed : Flag<["--"], "no-add-needed">;
def no_dynamic_linker : Flag<["--"], "no-dynamic-linker">;
def no_fatal_warnings : Flag<["--"], "no-fatal-warnings">;
def no_warn_common : Flag<["--", "-"], "no-warn-common">;
def no_warn_mismatch : Flag<["--"], "no-warn-mismatch">;
def rpath_link : Separate<["--", "-"], "rpath-link">;
def rpath_link_eq : Joined<["--", "-"], "rpath-link=">;
def version_script : Separate<["--"], "version-script">;
def warn_execstack : Flag<["--"], "warn-execstack">;
def warn_shared_textrel : Flag<["--"], "warn-shared-textrel">;
def G : Separate<["-"], "G">;

// Aliases for ignored options
def alias_version_script_version_script : Joined<["--"], "version-script=">, Alias<version_script>;

// LTO-related options.
def lto_no_discard_value_names : Flag<["-"], "lto-no-discard-value-names">;
def lto_jobs : Joined<["--"], "lto-jobs=">,
  HelpText<"Number of threads to run codegen">;
def disable_verify : Flag<["-"], "disable-verify">;
def mllvm : Separate<["-"], "mllvm">;
def save_temps : Flag<["-"], "save-temps">;
