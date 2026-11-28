include "llvm/Option/OptParser.td"

class F<string name, string help> : Flag<["-", "--"], name>, HelpText<help>;
multiclass Eq<string name, string meta, string help> {
  def _EQ : Joined<["-", "--"], name # "=">, MetaVarName<meta>, HelpText<help>;
  def : Separate<["-", "--"], name>, Alias<!cast<Option>(NAME # "_EQ")>;
}

def help : F<"help", "Display available options">;
def : F<"h", "Alias for --help">, Alias<help>;
def version : F<"version", "Display the version">;
def verbose : F<"verbose", "Enable verbose output">;

defm only : Eq<"only", "<vendor>",
  "Restrict to vendor: all (default), amdgpu, nvptx, intel">;
defm hip_api_version : Eq<"hip-api-version", "<version>",
  "HIP API version for device properties: auto (default), r0600, r0000, "
  "unversioned">;
