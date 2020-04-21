include "llvm/Option/OptParser.td"

def L: JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Add directory to library search path">;

def Z: Flag<["-"], "Z">,
  HelpText<"Do not add standard directories to library search path">;

def arch: Separate<["-"], "arch">, MetaVarName<"<arch-name>">,
  HelpText<"Architecture to link">;

def e: Separate<["-"], "e">, HelpText<"Name of entry point symbol">;

def l: Joined<["-"], "l">, MetaVarName<"<libname>">,
  HelpText<"Base name of library searched for in -L directories">;

def o: Separate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;

def v: Flag<["-"], "v">, HelpText<"Display the version number and exit">;

// Ignored options
def: Flag<["-"], "demangle">;
def: Flag<["-"], "dynamic">;
def: Flag<["-"], "no_deduplicate">;
def: Separate<["-"], "lto_library">;
def: Separate<["-"], "macosx_version_min">;
