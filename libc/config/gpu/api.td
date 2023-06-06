include "config/public_api.td"

include "spec/stdc.td"

def StringAPI : PublicAPI<"string.h"> {
  let Types = ["size_t"];
}

def StdlibAPI : PublicAPI<"stdlib.h"> {
  let Types = [
    "size_t",
    "__atexithandler_t",
  ];
}

def FenvAPI: PublicAPI<"fenv.h"> {
  let Types = ["fenv_t"];
}

def StdIOAPI : PublicAPI<"stdio.h"> {
  let Macros = [
    SimpleMacroDef<"_IOFBF", "0">,
    SimpleMacroDef<"_IOLBF", "1">,
    SimpleMacroDef<"_IONBF", "2">,
    SimpleMacroDef<"EOF", "-1">,
  ];
  let Types = ["size_t", "FILE"];
}
