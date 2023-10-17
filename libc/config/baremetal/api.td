include "config/public_api.td"

include "spec/stdc.td"

def CTypeAPI : PublicAPI<"ctype.h"> {
}

def FEnvAPI : PublicAPI<"fenv.h"> {
  let Types = ["fenv_t", "fexcept_t"];
}

def IntTypesAPI : PublicAPI<"inttypes.h"> {
  let Types = ["imaxdiv_t"];
}

def StdlibAPI : PublicAPI<"stdlib.h"> {
  let Types = [
    "div_t",
    "ldiv_t",
    "lldiv_t",
    "size_t",
    "__bsearchcompare_t",
    "__qsortcompare_t",
    "__atexithandler_t",
  ];
}

def StringAPI : PublicAPI<"string.h"> {
  let Types = ["size_t"];
}
