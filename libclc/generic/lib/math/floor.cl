#include <clc/clc.h>
#include "../clcmacro.h"

// Map the llvm intrinsic to an OpenCL function.
#define __CLC_FUNCTION __clc_floor
#define __CLC_INTRINSIC "llvm.floor"
#include "math/unary_intrin.inc"

#undef __CLC_FUNCTION
#define __CLC_FUNCTION floor
#include "unary_builtin.inc"
