include "llvm/Option/OptParser.td"

def o : JoinedOrSeparate<["-"], "o">,
        MetaVarName<"<path>">,
        HelpText<"Path to output SPIR-V binary">;
