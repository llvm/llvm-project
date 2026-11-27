include "llvm/Option/OptParser.td"

multiclass Eq<string name, string metavar, string help> {
  def NAME : Separate<["-", "--"], name>, MetaVarName<metavar>, HelpText<help>;
  def : Joined<["-", "--"], name #"=">, Alias<!cast<Separate>(NAME)>;
}

def help : Flag<["-", "--"], "help">, HelpText<"Display available options">;
defm o : Eq<"o", "<path>", "Output path for the merged YAML.">;
defm sarif_conflicts_out : Eq<"sarif-conflicts-out", "<path>",
  "Optional path. When supplied, write a SARIF document listing conflict "
  "clusters dropped from the merged output.">;
