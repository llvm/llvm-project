#include <clc/clc.h>

// Map the llvm intrinsic to an OpenCL function.
#define __CLC_FUNCTION __clc_rint
#define __CLC_INTRINSIC "llvm.rint"
#include "math/unary_intrin.inc"

#undef __CLC_FUNCTION
#define __CLC_FUNCTION rint
#include "unary_builtin.inc"
