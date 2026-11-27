include "llvm/Option/OptParser.td"

class F<string name, string help> : Flag<["-", "--"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
  def NAME : Separate<["-", "--"], name>, HelpText<help>;
  def : Joined<["-", "--"], name #"=">, Alias<!cast<Separate>(NAME)>;
}

def dump_imported_module_files : F<"dump-imported-module-files", "Print symbols and input files from imported modules">;
defm fmodule_format : Eq<"fmodule-format", "Container format for clang modules and PCH, 'raw' or 'obj'">, MetaVarName<"<format>">;
def help : F<"help", "Display available options">;
def ignore_macros : F<"ignore-macros", "Skip indexing macros">;
def include_locals : F<"include-locals", "Print local symbols">;
defm module_file : Eq<"module-file", "Path to module file to print symbols from">, MetaVarName<"<file>">;
def print_source_symbols : F<"print-source-symbols", "Print symbols from source">;
