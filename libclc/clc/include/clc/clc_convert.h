//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_CLC_CONVERT_H__
#define __CLC_CLC_CONVERT_H__

#include <clc/internal/clc.h>

#define _CLC_CONVERT_DECL(FROM_TYPE, TO_TYPE, SUFFIX)                          \
  _CLC_OVERLOAD _CLC_DECL TO_TYPE __clc_convert_##TO_TYPE##SUFFIX(FROM_TYPE x);

#define _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, TO_TYPE, SUFFIX)                   \
  _CLC_CONVERT_DECL(FROM_TYPE, TO_TYPE, SUFFIX)                                \
  _CLC_CONVERT_DECL(FROM_TYPE##2, TO_TYPE##2, SUFFIX)                          \
  _CLC_CONVERT_DECL(FROM_TYPE##3, TO_TYPE##3, SUFFIX)                          \
  _CLC_CONVERT_DECL(FROM_TYPE##4, TO_TYPE##4, SUFFIX)                          \
  _CLC_CONVERT_DECL(FROM_TYPE##8, TO_TYPE##8, SUFFIX)                          \
  _CLC_CONVERT_DECL(FROM_TYPE##16, TO_TYPE##16, SUFFIX)

#define _CLC_VECTOR_CONVERT_FROM1(FROM_TYPE, SUFFIX)                           \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, char, SUFFIX)                            \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, uchar, SUFFIX)                           \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, int, SUFFIX)                             \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, uint, SUFFIX)                            \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, short, SUFFIX)                           \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, ushort, SUFFIX)                          \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, long, SUFFIX)                            \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, ulong, SUFFIX)                           \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, float, SUFFIX)

#if defined(cl_khr_fp64) && defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define _CLC_VECTOR_CONVERT_FROM(FROM_TYPE, SUFFIX)                            \
  _CLC_VECTOR_CONVERT_FROM1(FROM_TYPE, SUFFIX)                                 \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, double, SUFFIX)                          \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, half, SUFFIX)
#elif defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define _CLC_VECTOR_CONVERT_FROM(FROM_TYPE, SUFFIX)                            \
  _CLC_VECTOR_CONVERT_FROM1(FROM_TYPE, SUFFIX)                                 \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, double, SUFFIX)
#elif defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _CLC_VECTOR_CONVERT_FROM(FROM_TYPE, SUFFIX)                            \
  _CLC_VECTOR_CONVERT_FROM1(FROM_TYPE, SUFFIX)                                 \
  _CLC_VECTOR_CONVERT_DECL(FROM_TYPE, half, SUFFIX)
#else
#define _CLC_VECTOR_CONVERT_FROM(FROM_TYPE, SUFFIX)                            \
  _CLC_VECTOR_CONVERT_FROM1(FROM_TYPE, SUFFIX)
#endif

#define _CLC_VECTOR_CONVERT_TO1(SUFFIX)                                        \
  _CLC_VECTOR_CONVERT_FROM(char, SUFFIX)                                       \
  _CLC_VECTOR_CONVERT_FROM(uchar, SUFFIX)                                      \
  _CLC_VECTOR_CONVERT_FROM(int, SUFFIX)                                        \
  _CLC_VECTOR_CONVERT_FROM(uint, SUFFIX)                                       \
  _CLC_VECTOR_CONVERT_FROM(short, SUFFIX)                                      \
  _CLC_VECTOR_CONVERT_FROM(ushort, SUFFIX)                                     \
  _CLC_VECTOR_CONVERT_FROM(long, SUFFIX)                                       \
  _CLC_VECTOR_CONVERT_FROM(ulong, SUFFIX)                                      \
  _CLC_VECTOR_CONVERT_FROM(float, SUFFIX)

#if defined(cl_khr_fp64) && defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define _CLC_VECTOR_CONVERT_TO(SUFFIX)                                         \
  _CLC_VECTOR_CONVERT_TO1(SUFFIX)                                              \
  _CLC_VECTOR_CONVERT_FROM(double, SUFFIX)                                     \
  _CLC_VECTOR_CONVERT_FROM(half, SUFFIX)
#elif defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define _CLC_VECTOR_CONVERT_TO(SUFFIX)                                         \
  _CLC_VECTOR_CONVERT_TO1(SUFFIX)                                              \
  _CLC_VECTOR_CONVERT_FROM(double, SUFFIX)
#elif defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define _CLC_VECTOR_CONVERT_TO(SUFFIX)                                         \
  _CLC_VECTOR_CONVERT_TO1(SUFFIX)                                              \
  _CLC_VECTOR_CONVERT_FROM(half, SUFFIX)
#else
#define _CLC_VECTOR_CONVERT_TO(SUFFIX) _CLC_VECTOR_CONVERT_TO1(SUFFIX)
#endif

#define _CLC_VECTOR_CONVERT_TO_SUFFIX(ROUND)                                   \
  _CLC_VECTOR_CONVERT_TO(_sat##ROUND)                                          \
  _CLC_VECTOR_CONVERT_TO(ROUND)

_CLC_VECTOR_CONVERT_TO_SUFFIX(_rtn)
_CLC_VECTOR_CONVERT_TO_SUFFIX(_rte)
_CLC_VECTOR_CONVERT_TO_SUFFIX(_rtz)
_CLC_VECTOR_CONVERT_TO_SUFFIX(_rtp)
_CLC_VECTOR_CONVERT_TO_SUFFIX()

#undef _CLC_VECTOR_CONVERT_TO_SUFFIX
#undef _CLC_VECTOR_CONVERT_TO
#undef _CLC_VECTOR_CONVERT_TO1
#undef _CLC_VECTOR_CONVERT_FROM
#undef _CLC_VECTOR_CONVERT_FROM1
#undef _CLC_VECTOR_CONVERT_DECL
#undef _CLC_CONVERT_DECL

#endif // __CLC_CLC_CONVERT_H__
