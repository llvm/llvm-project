_CLC_DEF _CLC_OVERLOAD float __clc_ldexp(float, int);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_DEF _CLC_OVERLOAD double __clc_ldexp(double, int);
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_DEF _CLC_OVERLOAD half __clc_ldexp(half, int);
#endif
