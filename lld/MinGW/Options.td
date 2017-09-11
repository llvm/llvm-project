include "llvm/Option/OptParser.td"

class F<string name>: Flag<["--", "-"], name>;
class J<string name>: Joined<["--", "-"], name>;
class S<string name>: Separate<["--", "-"], name>;

def L: JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Add a directory to the library search path">;
def entry: S<"entry">, MetaVarName<"<entry>">,
  HelpText<"Name of entry point symbol">;
def m: JoinedOrSeparate<["-"], "m">, HelpText<"Set target emulation">;
def o: JoinedOrSeparate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;
def l: JoinedOrSeparate<["-"], "l">, MetaVarName<"<libName>">,
  HelpText<"Root name of library to use">;
def shared: F<"shared">, HelpText<"Build a shared object">;
def subs: Separate<["--"], "subsystem">, HelpText<"Specify subsystem">;
def stack: Separate<["--"], "stack">;
def outlib: Separate<["--"], "out-implib">, HelpText<"Import library name">;
def verbose: F<"verbose">, HelpText<"Verbose mode">;

// LLD specific options
def _HASH_HASH_HASH : Flag<["-"], "###">,
    HelpText<"Print (but do not run) the commands to run for this compilation">;

// Currently stubs to avoid errors
def Bdynamic: F<"Bdynamic">, HelpText<"Link against shared libraries">;
def Bstatic: F<"Bstatic">, HelpText<"Do not link against shared libraries">;
def major_image_version: Separate<["--"], "major-image-version">;
def minor_image_version: Separate<["--"], "minor-image-version">;
def enable_auto_image_base: Flag<["--"], "enable-auto-image-base">;
def full_shutdown: Flag<["--"], "full-shutdown">;
def O: Joined<["-"], "O">, HelpText<"Optimize output file size">;
def sysroot: J<"sysroot">, HelpText<"Sysroot">;
def v: Flag<["-"], "v">, HelpText<"Display the version number">;
def version: F<"version">, HelpText<"Display the version number and exit">;

// Alias
def alias_entry_e: JoinedOrSeparate<["-"], "e">, Alias<entry>;
