include "llvm/Option/OptParser.td"

def L: JoinedOrSeparate<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Add directory to library search path">;

def Z: Flag<["-"], "Z">,
  HelpText<"Do not add standard directories to library search path">;

def arch: Separate<["-"], "arch">, MetaVarName<"<arch-name>">,
  HelpText<"Architecture to link">;

def dylib: Flag<["-"], "dylib">, HelpText<"Emit a shared library">;

def e: Separate<["-"], "e">, HelpText<"Name of entry point symbol">;

def install_name: Separate<["-"], "install_name">,
  MetaVarName<"<install-name>">,
  HelpText<"Set the install path of the dynamic library.">;

def l: Joined<["-"], "l">, MetaVarName<"<libname>">,
  HelpText<"Base name of library searched for in -L directories">;

def o: Separate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;

def order_file: Separate<["-"], "order_file">,
  HelpText<"Lay out symbols within each section in the order specified by the "
           "order file">;

def sub_library: Separate<["-"], "sub_library">, MetaVarName<"<libname>">,
  HelpText<"Re-export the specified dylib">;

def v: Flag<["-"], "v">, HelpText<"Display the version number and exit">;

// Ignored options
def: Flag<["-"], "demangle">;
def: Flag<["-"], "dynamic">;
def: Flag<["-"], "no_deduplicate">;
def platform_version: Flag<["-"], "platform_version">;
def: Separate<["-"], "lto_library">;
def: Separate<["-"], "macosx_version_min">;
def: Separate<["-"], "syslibroot">;
