include "config/public_api.td"

include "spec/stdc.td"
include "spec/posix.td"
include "spec/gpu_ext.td"
include "spec/gnu_ext.td"
include "spec/stdc_ext.td"
include "spec/llvm_libc_ext.td"


def StringAPI : PublicAPI<"string.h"> {
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
    "__qsortrcompare_t",
    "__atexithandler_t",
  ];
}

def FenvAPI: PublicAPI<"fenv.h"> {
  let Types = ["fenv_t"];
}

def StdIOAPI : PublicAPI<"stdio.h"> {
  let Types = [
    "FILE",
    "off_t",
    "size_t",
  ];
}

def IntTypesAPI : PublicAPI<"inttypes.h"> {
  let Types = ["imaxdiv_t"];
}

def TimeAPI : PublicAPI<"time.h"> {
  let Types = [
    "clock_t",
    "time_t",
    "struct timespec",
  ];
}
