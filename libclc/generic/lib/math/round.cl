#include <clc/clc.h>

// Map the llvm intrinsic to an OpenCL function.
#define __CLC_FUNCTION __clc_round
#define __CLC_INTRINSIC "llvm.round"
#include "math/unary_intrin.inc"

#undef __CLC_FUNCTION
#define __CLC_FUNCTION round
#include "unary_builtin.inc"
