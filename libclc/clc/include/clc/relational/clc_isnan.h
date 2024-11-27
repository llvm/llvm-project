#ifndef __CLC_RELATIONAL_CLC_ISNAN_H__
#define __CLC_RELATIONAL_CLC_ISNAN_H__

#if defined(CLC_CLSPV) || defined(CLC_SPIRV)
// clspv and spir-v targets provide their own OpenCL-compatible isnan
#define __clc_isnan isnan
#else

#include <clc/clcfunc.h>

#define _CLC_ISNAN_DECL(RET_TYPE, ARG_TYPE)                                    \
  _CLC_OVERLOAD _CLC_DECL RET_TYPE __clc_isnan(ARG_TYPE);

#define _CLC_VECTOR_ISNAN_DECL(RET_TYPE, ARG_TYPE)                             \
  _CLC_ISNAN_DECL(RET_TYPE##2, ARG_TYPE##2)                                    \
  _CLC_ISNAN_DECL(RET_TYPE##3, ARG_TYPE##3)                                    \
  _CLC_ISNAN_DECL(RET_TYPE##4, ARG_TYPE##4)                                    \
  _CLC_ISNAN_DECL(RET_TYPE##8, ARG_TYPE##8)                                    \
  _CLC_ISNAN_DECL(RET_TYPE##16, ARG_TYPE##16)

_CLC_ISNAN_DECL(int, float)
_CLC_VECTOR_ISNAN_DECL(int, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_ISNAN_DECL(int, double)
_CLC_VECTOR_ISNAN_DECL(long, double)
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_ISNAN_DECL(int, half)
_CLC_VECTOR_ISNAN_DECL(short, half)
#endif

#undef _CLC_ISNAN_DECL
#undef _CLC_VECTOR_ISNAN_DECL

#endif

#endif // __CLC_RELATIONAL_CLC_ISNAN_H__
