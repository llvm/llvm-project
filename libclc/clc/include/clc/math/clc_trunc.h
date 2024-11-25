#ifndef __CLC_MATH_CLC_TRUNC_H__
#define __CLC_MATH_CLC_TRUNC_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible trunc
#define __clc_trunc trunc
#else

// Map the function to an LLVM intrinsic
#define __CLC_FUNCTION __clc_trunc
#define __CLC_INTRINSIC "llvm.trunc"
#include <clc/math/unary_intrin.inc>

#undef __CLC_INTRINSIC
#undef __CLC_FUNCTION

#endif

#endif // __CLC_MATH_CLC_TRUNC_H__
