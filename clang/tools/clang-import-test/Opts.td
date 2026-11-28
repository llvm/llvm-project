include "llvm/Option/OptParser.td"

class F<string name, string help> : Flag<["-", "--"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
  def NAME : Separate<["-", "--"], name>, HelpText<help>;
  def : Joined<["-", "--"], name #"=">, Alias<!cast<Separate>(NAME)>;
}

def direct : F<"direct", "Use the parsed declarations without indirection">;
def dump_ast : F<"dump-ast", "Dump combined AST">;
def dump_ir : F<"dump-ir", "Dump IR from final parse">;
defm expression : Eq<"expression", "Path to a file containing the expression to parse">, MetaVarName<"<file>">;
def help : F<"help", "Display available options">;
defm import : Eq<"import", "Path to a file containing declarations to import">, MetaVarName<"<file>">;
def objc_arc : F<"objc-arc", "Emable ObjC ARC">;
def use_origins : F<"use-origins", "Use DeclContext origin information for more accurate lookups">;
defm x : Eq<"x", "The language to parse (default: c++)">, MetaVarName<"<language>">;
defm Xcc : Eq<"Xcc", "Argument to pass to the CompilerInvocation">, MetaVarName<"<arg>">;
