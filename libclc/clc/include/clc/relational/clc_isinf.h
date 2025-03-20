#ifndef __CLC_RELATIONAL_CLC_ISINF_H__
#define __CLC_RELATIONAL_CLC_ISINF_H__

#include <clc/clcfunc.h>
#include <clc/clctypes.h>

#define _CLC_ISINF_DECL(RET_TYPE, ARG_TYPE)                                    \
  _CLC_OVERLOAD _CLC_DECL RET_TYPE __clc_isinf(ARG_TYPE);

#define _CLC_VECTOR_ISINF_DECL(RET_TYPE, ARG_TYPE)                             \
  _CLC_ISINF_DECL(RET_TYPE##2, ARG_TYPE##2)                                    \
  _CLC_ISINF_DECL(RET_TYPE##3, ARG_TYPE##3)                                    \
  _CLC_ISINF_DECL(RET_TYPE##4, ARG_TYPE##4)                                    \
  _CLC_ISINF_DECL(RET_TYPE##8, ARG_TYPE##8)                                    \
  _CLC_ISINF_DECL(RET_TYPE##16, ARG_TYPE##16)

_CLC_ISINF_DECL(int, float)
_CLC_VECTOR_ISINF_DECL(int, float)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_ISINF_DECL(int, double)
_CLC_VECTOR_ISINF_DECL(long, double)
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_ISINF_DECL(int, half)
_CLC_VECTOR_ISINF_DECL(short, half)
#endif

#undef _CLC_ISINF_DECL
#undef _CLC_VECTOR_ISINF_DECL

#endif // __CLC_RELATIONAL_CLC_ISINF_H__
