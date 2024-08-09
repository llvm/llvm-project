include "config/public_api.td"

include "spec/stdc.td"
include "spec/stdc_ext.td"
include "spec/bsd_ext.td"
include "spec/llvm_libc_stdfix_ext.td"


def CTypeAPI : PublicAPI<"ctype.h"> {
}

def FEnvAPI : PublicAPI<"fenv.h"> {
  let Types = ["fenv_t", "fexcept_t"];
}

def IntTypesAPI : PublicAPI<"inttypes.h"> {
  let Types = ["imaxdiv_t"];
}

def MathAPI : PublicAPI<"math.h"> {
  let Types = ["double_t", "float_t"];
}

def StdIOAPI : PublicAPI<"stdio.h"> {
  let Types = ["size_t"];
}

def StdlibAPI : PublicAPI<"stdlib.h"> {
  let Types = [
    "div_t",
    "ldiv_t",
    "lldiv_t",
    "size_t",
    "__bsearchcompare_t",
    "__qsortcompare_t",
  ];
}

def StringAPI : PublicAPI<"string.h"> {
  let Types = ["size_t"];
}

def TimeAPI : PublicAPI<"time.h"> {
  let Types = [
    "clock_t",
    "time_t",
    "struct tm",
    "struct timespec",
  ];
}

def UCharAPI : PublicAPI<"uchar.h"> {
  let Types = [
    "mbstate_t",
    "char8_t",
    "char16_t",
    "char32_t",
  ];
}
