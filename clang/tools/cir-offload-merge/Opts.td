include "llvm/Option/OptParser.td"

def combine : Flag<["-", "--"], "combine">, HelpText<"Combine CIR inputs">;
def help : Flag<["-", "--"], "help">, HelpText<"Display available options">;
def o : Separate<["-", "--"], "o">, MetaVarName<"<filename>">,
  HelpText<"Output CIR file">;
def : Joined<["-", "--"], "o=">, Alias<o>;
