include "llvm/Option/OptParser.td"
include "clang/Tooling/CommonOptionsParser.td"

class F<string name, string help> : Flag<["--", "-"], name>, HelpText<help>;
class Short<string name, string help> : Flag<["--", "-"], name>, HelpText<help>;

def help : Flag<["--"], "help">, HelpText<"Display this help">;
def version : Flag<["--"], "version">, HelpText<"Display the version">;

defm offset : Eq<"offset",
                 "Locates the symbol by offset as opposed to <line>:<column>.">;

def inplace : Flag<["-", "--"], "i">, HelpText<"Overwrite edited <file>s.">;

defm qualified_name
    : Eq<"qualified-name", "The fully qualified name of the symbol.">;

defm new_name : Eq<"new-name", "The new name to change the symbol to.">;

def print_name
    : F<"pn", "Print the found symbol's name prior to renaming to stderr.">;

def print_locations
    : F<"pl", "Print the locations affected by renaming to stderr.">;

defm export_fixes
    : Eq<"export-fixes", "YAML file to store suggested fixes in.">,
      MetaVarName<"<filename>">;

defm input : Eq<"input", "YAML file to load oldname-newname pairs from.">;

def force : F<"force", "Ignore nonexistent qualified names.">;
