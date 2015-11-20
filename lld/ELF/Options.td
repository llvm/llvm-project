include "llvm/Option/OptParser.td"

def Bsymbolic: Flag<["-"], "Bsymbolic">,
  HelpText<"Bind defined symbols locally">;

def Bdynamic: Flag<["-"], "Bdynamic">,
  HelpText<"Link against shared libraries">;

def Bstatic: Flag<["-"], "Bstatic">,
  HelpText<"Do not link against shared libraries">;

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

def enable_new_dtags : Flag<["--"], "enable-new-dtags">,
  HelpText<"Enable new dynamic tags">;

def entry : Separate<["--", "-"], "entry">, MetaVarName<"<entry>">,
  HelpText<"Name of entry point symbol">;

def export_dynamic : Flag<["--", "-"], "export-dynamic">,
  HelpText<"Put symbols in the dynamic symbol table">;

def fini : Separate<["-"], "fini">, MetaVarName<"<symbol>">,
  HelpText<"Specify a finalizer function">;

def hash_style : Separate<["--", "-"], "hash-style">,
  HelpText<"Specify hash style (sysv, gnu or both)">;

def gc_sections : Flag<["--"], "gc-sections">,
  HelpText<"Enable garbage collection of unused sections">;

def init : Separate<["-"], "init">, MetaVarName<"<symbol>">,
  HelpText<"Specify an initializer function">;

def l : JoinedOrSeparate<["-"], "l">, MetaVarName<"<libName>">,
  HelpText<"Root name of library to use">;

def m : JoinedOrSeparate<["-"], "m">,
  HelpText<"Set target emulation">;

def no_allow_shlib_undefined : Flag<["--"], "no-allow-shlib-undefined">;

def no_as_needed : Flag<["--"], "no-as-needed">;

def no_whole_archive : Flag<["--", "-"], "no-whole-archive">,
  HelpText<"Restores the default behavior of loading archive members">;

def noinhibit_exec : Flag<["--"], "noinhibit-exec">,
  HelpText<"Retain the executable output file whenever it is still usable">;

def no_undefined : Flag<["--"], "no-undefined">,
  HelpText<"Report unresolved symbols even if the linker is creating a shared library">;

def o : Separate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;

def rpath : Separate<["-"], "rpath">,
  HelpText<"Add a DT_RUNPATH to the output">;

def relocatable : Flag<["--"], "relocatable">;

def script : Separate<["--"], "script">, HelpText<"Read linker script">;

def shared : Flag<["-"], "shared">,
  HelpText<"Build a shared object">;

def soname : Joined<["-"], "soname=">,
  HelpText<"Set DT_SONAME">;

def strip_all : Flag<["--"], "strip-all">,
  HelpText<"Strip all symbols">;

def sysroot : Joined<["--"], "sysroot=">,
  HelpText<"Set the system root">;

def undefined : Joined<["--"], "undefined=">,
  HelpText<"Force undefined symbol during linking">;

def verbose : Flag<["--"], "verbose">;

def whole_archive : Flag<["--", "-"], "whole-archive">,
  HelpText<"Force load of all members in a static library">;

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
def alias_entry_e : Separate<["-"], "e">, Alias<entry>;
def alias_export_dynamic_E: Flag<["-"], "E">, Alias<export_dynamic>;
def alias_fini_fini : Joined<["-"], "fini=">, Alias<fini>;
def alias_hash_style_hash_style : Joined<["--", "-"], "hash-style=">, Alias<hash_style>;
def alias_init_init : Joined<["-"], "init=">, Alias<init>;
def alias_l__library : Joined<["--"], "library=">, Alias<l>;
def alias_o_output : Joined<["--"], "output=">, Alias<o>;
def alias_rpath_rpath : Joined<["-"], "rpath=">, Alias<rpath>;
def alias_relocatable_r : Flag<["-"], "r">, Alias<relocatable>;
def alias_shared_Bshareable : Flag<["-"], "Bshareable">, Alias<shared>;
def alias_soname_h : Separate<["-"], "h">, Alias<soname>;
def alias_soname_soname : Separate<["-"], "soname">, Alias<soname>;
def alias_script_T : Separate<["-"], "T">, Alias<script>;
def alias_strip_all: Flag<["-"], "s">, Alias<strip_all>;
def alias_undefined_u : Separate<["-"], "u">, Alias<undefined>;

// Our symbol resolution algorithm handles symbols in archive files differently
// than traditional linkers, so we don't need --start-group and --end-group.
// These options are recongized for compatibility but ignored.
def end_group : Flag<["--"], "end-group">;
def end_group_paren: Flag<["-"], ")">;
def start_group : Flag<["--"], "start-group">;
def start_group_paren: Flag<["-"], "(">;

// Options listed below are silently ignored for now for compatibility.
def build_id : Flag<["--"], "build-id">;
def eh_frame_hdr : Flag<["--"], "eh-frame-hdr">;
def fatal_warnings : Flag<["--"], "fatal-warnings">;
def no_add_needed : Flag<["--"], "no-add-needed">;
def no_fatal_warnings : Flag<["--"], "no-fatal-warnings">;
def no_warn_mismatch : Flag<["--"], "no-warn-mismatch">;
def version_script : Separate<["--"], "version-script">;
def warn_common : Flag<["--"], "warn-common">;
def warn_shared_textrel : Flag<["--"], "warn-shared-textrel">;
def G : Separate<["-"], "G">;
