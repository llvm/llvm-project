//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/internal/clc.h>
#include <clc/relational/relational.h>

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC2(RET_TYPE, __CLC_FUNCTION, ARG_TYPE)  \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE __CLC_FUNCTION(ARG_TYPE x) {                 \
    return (RET_TYPE)((RET_TYPE){__CLC_FUNCTION(x.lo),                         \
                                 __CLC_FUNCTION(x.hi)} != (RET_TYPE)0);        \
  }

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC3(RET_TYPE, __CLC_FUNCTION, ARG_TYPE)  \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE __CLC_FUNCTION(ARG_TYPE x) {                 \
    return (RET_TYPE)((RET_TYPE){__CLC_FUNCTION(x.s0), __CLC_FUNCTION(x.s1),   \
                                 __CLC_FUNCTION(x.s2)} != (RET_TYPE)0);        \
  }

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC4(RET_TYPE, __CLC_FUNCTION, ARG_TYPE)  \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE __CLC_FUNCTION(ARG_TYPE x) {                 \
    return (RET_TYPE)((RET_TYPE){__CLC_FUNCTION(x.s0), __CLC_FUNCTION(x.s1),   \
                                 __CLC_FUNCTION(x.s2),                         \
                                 __CLC_FUNCTION(x.s3)} != (RET_TYPE)0);        \
  }

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC8(RET_TYPE, __CLC_FUNCTION, ARG_TYPE)  \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE __CLC_FUNCTION(ARG_TYPE x) {                 \
    return (RET_TYPE)((RET_TYPE){__CLC_FUNCTION(x.s0), __CLC_FUNCTION(x.s1),   \
                                 __CLC_FUNCTION(x.s2), __CLC_FUNCTION(x.s3),   \
                                 __CLC_FUNCTION(x.s4), __CLC_FUNCTION(x.s5),   \
                                 __CLC_FUNCTION(x.s6),                         \
                                 __CLC_FUNCTION(x.s7)} != (RET_TYPE)0);        \
  }

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC16(RET_TYPE, __CLC_FUNCTION, ARG_TYPE) \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE __CLC_FUNCTION(ARG_TYPE x) {                 \
    return (RET_TYPE)((RET_TYPE){__CLC_FUNCTION(x.s0), __CLC_FUNCTION(x.s1),   \
                                 __CLC_FUNCTION(x.s2), __CLC_FUNCTION(x.s3),   \
                                 __CLC_FUNCTION(x.s4), __CLC_FUNCTION(x.s5),   \
                                 __CLC_FUNCTION(x.s6), __CLC_FUNCTION(x.s7),   \
                                 __CLC_FUNCTION(x.s8), __CLC_FUNCTION(x.s9),   \
                                 __CLC_FUNCTION(x.sa), __CLC_FUNCTION(x.sb),   \
                                 __CLC_FUNCTION(x.sc), __CLC_FUNCTION(x.sd),   \
                                 __CLC_FUNCTION(x.se),                         \
                                 __CLC_FUNCTION(x.sf)} != (RET_TYPE)0);        \
  }

#define _CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(RET_TYPE, __CLC_FUNCTION,         \
                                             ARG_TYPE)                         \
  _CLC_DEFINE_RELATIONAL_UNARY_VEC2(RET_TYPE##2, __CLC_FUNCTION, ARG_TYPE##2)  \
  _CLC_DEFINE_RELATIONAL_UNARY_VEC3(RET_TYPE##3, __CLC_FUNCTION, ARG_TYPE##3)  \
  _CLC_DEFINE_RELATIONAL_UNARY_VEC4(RET_TYPE##4, __CLC_FUNCTION, ARG_TYPE##4)  \
  _CLC_DEFINE_RELATIONAL_UNARY_VEC8(RET_TYPE##8, __CLC_FUNCTION, ARG_TYPE##8)  \
  _CLC_DEFINE_RELATIONAL_UNARY_VEC16(RET_TYPE##16, __CLC_FUNCTION, ARG_TYPE##16)

_CLC_DEF _CLC_OVERLOAD int __clc_signbit(float x) {
  return __builtin_signbitf(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(int, __clc_signbit, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// The scalar version of __clc_signbit(double) returns an int, but the vector
// versions return long.

_CLC_DEF _CLC_OVERLOAD int __clc_signbit(double x) {
  return __builtin_signbit(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(long, __clc_signbit, double)

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// The scalar version of __clc_signbit(half) returns an int, but the vector
// versions return short.

_CLC_DEF _CLC_OVERLOAD int __clc_signbit(half x) {
  return __builtin_signbit(x);
}

_CLC_DEFINE_RELATIONAL_UNARY_VEC_ALL(short, __clc_signbit, half)

#endif
