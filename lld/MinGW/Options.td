include "llvm/Option/OptParser.td"

class F<string name>: Flag<["--", "-"], name>;
class J<string name>: Joined<["--", "-"], name>;
class S<string name>: Separate<["--", "-"], name>;

def L: JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Add a directory to the library search path">;
def entry: S<"entry">, MetaVarName<"<entry>">,
  HelpText<"Name of entry point symbol">;
def export_all_symbols: F<"export-all-symbols">,
    HelpText<"Export all symbols even if a def file or dllexport attributes are used">;
def l: JoinedOrSeparate<["-"], "l">, MetaVarName<"<libName>">,
  HelpText<"Root name of library to use">;
def m: JoinedOrSeparate<["-"], "m">, HelpText<"Set target emulation">;
def no_whole_archive: F<"no-whole-archive">,
    HelpText<"No longer include all object files for following archives">;
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
def whole_archive: F<"whole-archive">,
    HelpText<"Include all object files for following archives">;
def verbose: F<"verbose">, HelpText<"Verbose mode">;

// LLD specific options
def _HASH_HASH_HASH : Flag<["-"], "###">,
    HelpText<"Print (but do not run) the commands to run for this compilation">;
def mllvm: S<"mllvm">;
def Xlink : J<"Xlink=">, MetaVarName<"<arg>">,
    HelpText<"Pass <arg> to the COFF linker">;

// Currently stubs to avoid errors
def Bdynamic: F<"Bdynamic">, HelpText<"Link against shared libraries">;
def Bstatic: F<"Bstatic">, HelpText<"Do not link against shared libraries">;
def O: Joined<["-"], "O">, HelpText<"Optimize output file size">;
def enable_auto_image_base: F<"enable-auto-image-base">;
def enable_auto_import: F<"enable-auto-import">;
def full_shutdown: Flag<["--"], "full-shutdown">;
def major_image_version: S<"major-image-version">;
def minor_image_version: S<"minor-image-version">;
def sysroot: J<"sysroot">, HelpText<"Sysroot">;
def v: Flag<["-"], "v">, HelpText<"Display the version number">;
def version: F<"version">, HelpText<"Display the version number and exit">;

// Alias
def alias_entry_e: JoinedOrSeparate<["-"], "e">, Alias<entry>;
def alias_strip_s: Flag<["-"], "s">, Alias<strip_all>;
