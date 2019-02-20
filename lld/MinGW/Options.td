include "llvm/Option/OptParser.td"

class F<string name>: Flag<["--", "-"], name>;
class J<string name>: Joined<["--", "-"], name>;
class S<string name>: Separate<["--", "-"], name>;

def L: JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Add a directory to the library search path">;
def Bdynamic: F<"Bdynamic">, HelpText<"Link against shared libraries">;
def Bstatic: F<"Bstatic">, HelpText<"Do not link against shared libraries">;
def dynamicbase: F<"dynamicbase">, HelpText<"Enable ASLR">;
def entry: S<"entry">, MetaVarName<"<entry>">,
  HelpText<"Name of entry point symbol">;
def export_all_symbols: F<"export-all-symbols">,
    HelpText<"Export all symbols even if a def file or dllexport attributes are used">;
def gc_sections: F<"gc-sections">, HelpText<"Remove unused sections">;
def icf: J<"icf=">, HelpText<"Identical code folding">;
def image_base: S<"image-base">, HelpText<"Base address of the program">;
def kill_at: F<"kill-at">, HelpText<"Remove @n from exported symbols">;
def l: JoinedOrSeparate<["-"], "l">, MetaVarName<"<libName>">,
  HelpText<"Root name of library to use">;
def m: JoinedOrSeparate<["-"], "m">, HelpText<"Set target emulation">;
def map: S<"Map">, HelpText<"Output a linker map">;
def map_eq: J<"Map=">, Alias<map>;
def no_whole_archive: F<"no-whole-archive">,
    HelpText<"No longer include all object files for following archives">;
def large_address_aware: Flag<["--"], "large-address-aware">,
    HelpText<"Enable large addresses">;
def no_gc_sections: F<"no-gc-sections">, HelpText<"Don't remove unused sections">;
def o: JoinedOrSeparate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;
def out_implib: Separate<["--"], "out-implib">, HelpText<"Import library name">;
def out_implib_eq: Joined<["--"], "out-implib=">, Alias<out_implib>;
def output_def: S<"output-def">, HelpText<"Output def file">;
def shared: F<"shared">, HelpText<"Build a shared object">;
def subs: S<"subsystem">, HelpText<"Specify subsystem">;
def stack: S<"stack">;
def strip_all: F<"strip-all">,
    HelpText<"Omit all symbol information from the output binary">;
def strip_debug: F<"strip-debug">,
    HelpText<"Omit all debug information, but keep symbol information">;
def whole_archive: F<"whole-archive">,
    HelpText<"Include all object files for following archives">;
def verbose: F<"verbose">, HelpText<"Verbose mode">;
def require_defined: S<"require-defined">,
    HelpText<"Force symbol to be added to symbol table as an undefined one">;
def require_defined_eq: J<"require-defined=">, Alias<require_defined>;

// LLD specific options
def _HASH_HASH_HASH : Flag<["-"], "###">,
    HelpText<"Print (but do not run) the commands to run for this compilation">;
def mllvm: S<"mllvm">;
def pdb: S<"pdb">, HelpText<"Specify output PDB debug information file">;
def Xlink : J<"Xlink=">, MetaVarName<"<arg>">,
    HelpText<"Pass <arg> to the COFF linker">;

// Alias
def alias_entry_e: JoinedOrSeparate<["-"], "e">, Alias<entry>;
def alias_strip_s: Flag<["-"], "s">, Alias<strip_all>;
def alias_strip_S: Flag<["-"], "S">, Alias<strip_debug>;

// Ignored options
def: Joined<["-"], "O">;
def: F<"build-id">;
def: F<"disable-auto-image-base">;
def: F<"enable-auto-image-base">;
def: F<"enable-auto-import">;
def: F<"end-group">;
def: Flag<["--"], "full-shutdown">;
def: F<"high-entropy-va">;
def: S<"major-image-version">;
def: S<"minor-image-version">;
def: F<"no-seh">;
def: F<"nxcompat">;
def: F<"pic-executable">;
def: S<"plugin">;
def: J<"plugin=">;
def: S<"plugin-opt">;
def: J<"plugin-opt=">;
def: J<"sysroot">;
def: F<"start-group">;
def: F<"tsaware">;
def: Flag<["-"], "v">;
def: F<"version">;
