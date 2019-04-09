#include <clc/clc.h>

// Map the llvm intrinsic to an OpenCL function.
#define __CLC_FUNCTION __clc_trunc
#define __CLC_INTRINSIC "llvm.trunc"
#include "math/unary_intrin.inc"

#undef __CLC_FUNCTION
#define __CLC_FUNCTION trunc
#include "unary_builtin.inc"
