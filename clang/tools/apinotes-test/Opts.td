include "llvm/Option/OptParser.td"

def help : Flag<["-", "--"], "help">, HelpText<"Display available options">;
def o : Separate<["-", "--"], "o">, MetaVarName<"<filename>">,
  HelpText<"Output filename">;
def : Joined<["-", "--"], "o=">, Alias<o>;
