#ifndef __CLC_MATH_CLC_CEIL_H__
#define __CLC_MATH_CLC_CEIL_H__

// Map the function to an LLVM intrinsic
#define __CLC_FUNCTION __clc_ceil
#define __CLC_INTRINSIC "llvm.ceil"
#include <clc/math/unary_intrin.inc>

#undef __CLC_INTRINSIC
#undef __CLC_FUNCTION

#endif // __CLC_MATH_CLC_CEIL_H__
