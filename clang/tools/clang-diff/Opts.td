include "llvm/Option/OptParser.td"

class F<string name, string help> : Flag<["-", "--"], name>, HelpText<help>;

multiclass Eq<string name, string help> {
  def NAME : Separate<["-", "--"], name>, HelpText<help>;
  def : Joined<["-", "--"], name #"=">, Alias<!cast<Separate>(NAME)>;
}

def ast_dump : F<"ast-dump", "Print the internal representation of the AST.">;
def ast_dump_json : F<"ast-dump-json", "Print the internal representation of the AST as JSON.">;
def dump_matches : F<"dump-matches", "Print the matched nodes.">;
defm extra_arg : Eq<"extra-arg", "Additional argument to append to the compiler command line">, MetaVarName<"<arg>">;
defm extra_arg_before : Eq<"extra-arg-before", "Additional argument to prepend to the compiler command line">, MetaVarName<"<arg>">;
def help : F<"help", "Display available options">;
def html : F<"html", "Output a side-by-side diff in HTML.">;
defm p : Eq<"p", "Build path">, MetaVarName<"<build-path>">;
defm s : Eq<"s", "Maximum subtree size for the optimal bottom-up mapping">, MetaVarName<"<maxsize>">;
defm stop_diff_after : Eq<"stop-diff-after", "Stop after the given matching phase">, MetaVarName<"<topdown|bottomup>">;
