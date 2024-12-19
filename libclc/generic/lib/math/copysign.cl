#include <clc/clc.h>
#include <clc/clcmacro.h>

_CLC_DEFINE_BINARY_BUILTIN(float, copysign, __builtin_copysignf, float, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, copysign, __builtin_copysign, double, double)

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half copysign(half x, half y)
{
   ushort sign_x = as_ushort(x) & 0x8000u;
   ushort unsigned_y = as_ushort(y) & 0x7ffffu;

   return as_half((ushort)(sign_x | unsigned_y));
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, copysign, half, half)

#endif
