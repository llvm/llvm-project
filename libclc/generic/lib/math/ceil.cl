#include <clc/clc.h>
#include "../clcmacro.h"

// Map the llvm intrinsic to an OpenCL function.
#define __CLC_FUNCTION __clc_ceil
#define __CLC_INTRINSIC "llvm.ceil"
#include "math/unary_intrin.inc"

#undef __CLC_FUNCTION
#define __CLC_FUNCTION ceil
#include "unary_builtin.inc"
