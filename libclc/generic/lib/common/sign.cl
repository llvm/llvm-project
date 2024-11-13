#include <clc/clc.h>
#include <clc/clcmacro.h>

#define SIGN(TYPE, F) \
_CLC_DEF _CLC_OVERLOAD TYPE sign(TYPE x) { \
  if (isnan(x)) { \
    return 0.0F;   \
  }               \
  if (x > 0.0F) { \
    return 1.0F;  \
  }               \
  if (x < 0.0F) { \
    return -1.0F; \
  }               \
  return x; /* -0.0 or +0.0 */  \
}

SIGN(float, f)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, sign, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

SIGN(double, )
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, sign, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

SIGN(half,)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, sign, half)

#endif
