#ifndef __CLC_MATH_CLC_RINT_H__
#define __CLC_MATH_CLC_RINT_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible rint
#define __clc_rint rint
#else

// Map the function to an LLVM intrinsic
#define __CLC_FUNCTION __clc_rint
#define __CLC_INTRINSIC "llvm.rint"
#include <clc/math/unary_intrin.inc>

#undef __CLC_INTRINSIC
#undef __CLC_FUNCTION

#endif

#endif // __CLC_MATH_CLC_RINT_H__
