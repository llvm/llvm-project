//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/utils.h>

#ifdef __CLC_SCALAR

#ifndef __CLC_MIN_VECSIZE
#define __CLC_MIN_VECSIZE 2
#endif

#ifndef __CLC_IMPL_FUNCTION
#define __CLC_IMPL_FUNCTION __CLC_FUNCTION
#endif

#ifndef __CLC_DEF_SPEC
#define __CLC_DEF_SPEC _CLC_DEF
#endif

#ifndef __CLC_RET_TYPE
#define __CLC_RET_TYPE __CLC_GENTYPE
#endif

#ifndef __CLC_ARG1_TYPE
#define __CLC_ARG1_TYPE __CLC_GENTYPE
#endif

#ifndef __CLC_ARG2_TYPE
#define __CLC_ARG2_TYPE __CLC_GENTYPE
#endif

#ifndef __CLC_ARG3_TYPE
#define __CLC_ARG3_TYPE __CLC_GENTYPE
#endif

#define __CLC_RET_VECTYPE __CLC_XCONCAT(__CLC_RET_TYPE, __CLC_VECTOR_SIZE)
#define __CLC_ARG1_VECTYPE __CLC_XCONCAT(__CLC_ARG1_TYPE, __CLC_VECTOR_SIZE)
#define __CLC_ARG2_VECTYPE __CLC_XCONCAT(__CLC_ARG2_TYPE, __CLC_VECTOR_SIZE)
#define __CLC_ARG3_VECTYPE __CLC_XCONCAT(__CLC_ARG3_TYPE, __CLC_VECTOR_SIZE)

#if __CLC_MIN_VECSIZE == 1
_CLC_OVERLOAD __CLC_DEF_SPEC __CLC_RET_TYPE __CLC_FUNCTION(__CLC_ARG1_TYPE x,
                                                           __CLC_ARG2_TYPE y,
                                                           __CLC_ARG3_TYPE z) {
  return __CLC_IMPL_FUNCTION(x, y, z);
}
#endif // __CLC_MIN_VECSIZE == 1

#if __CLC_MIN_VECSIZE <= 2
#define __CLC_VECTOR_SIZE 2
_CLC_OVERLOAD __CLC_DEF_SPEC __CLC_RET_VECTYPE
__CLC_FUNCTION(__CLC_ARG1_VECTYPE x, __CLC_ARG2_VECTYPE y, __CLC_ARG3_VECTYPE z) {
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0, y.s0, z.s0),
                           __CLC_IMPL_FUNCTION(x.s1, y.s1, z.s1));
}
#undef __CLC_VECTOR_SIZE
#endif // __CLC_MIN_VECSIZE <= 2

#if __CLC_MIN_VECSIZE > 3
#error "__CLC_MIN_VECSIZE > 3 isn't implemented"
#endif

#define __CLC_VECTOR_SIZE 3
_CLC_OVERLOAD __CLC_DEF_SPEC __CLC_RET_VECTYPE
__CLC_FUNCTION(__CLC_ARG1_VECTYPE x, __CLC_ARG2_VECTYPE y, __CLC_ARG3_VECTYPE z) {
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0, y.s0, z.s0),
                           __CLC_IMPL_FUNCTION(x.s1, y.s1, z.s1),
                           __CLC_IMPL_FUNCTION(x.s2, y.s2, z.s2));
}
#undef __CLC_VECTOR_SIZE

#define __CLC_VECTOR_SIZE 4
_CLC_OVERLOAD __CLC_DEF_SPEC __CLC_RET_VECTYPE
__CLC_FUNCTION(__CLC_ARG1_VECTYPE x, __CLC_ARG2_VECTYPE y, __CLC_ARG3_VECTYPE z) {
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0, y.s0, z.s0),
                           __CLC_IMPL_FUNCTION(x.s1, y.s1, z.s1),
                           __CLC_IMPL_FUNCTION(x.s2, y.s2, z.s2),
                           __CLC_IMPL_FUNCTION(x.s3, y.s3, z.s3));
}
#undef __CLC_VECTOR_SIZE

#define __CLC_VECTOR_SIZE 8
_CLC_OVERLOAD __CLC_DEF_SPEC __CLC_RET_VECTYPE
__CLC_FUNCTION(__CLC_ARG1_VECTYPE x, __CLC_ARG2_VECTYPE y, __CLC_ARG3_VECTYPE z) {
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0, y.s0, z.s0),
                           __CLC_IMPL_FUNCTION(x.s1, y.s1, z.s1),
                           __CLC_IMPL_FUNCTION(x.s2, y.s2, z.s2),
                           __CLC_IMPL_FUNCTION(x.s3, y.s3, z.s3),
                           __CLC_IMPL_FUNCTION(x.s4, y.s4, z.s4),
                           __CLC_IMPL_FUNCTION(x.s5, y.s5, z.s5),
                           __CLC_IMPL_FUNCTION(x.s6, y.s6, z.s6),
                           __CLC_IMPL_FUNCTION(x.s7, y.s7, z.s7));
}
#undef __CLC_VECTOR_SIZE

#define __CLC_VECTOR_SIZE 16
_CLC_OVERLOAD __CLC_DEF_SPEC __CLC_RET_VECTYPE
__CLC_FUNCTION(__CLC_ARG1_VECTYPE x, __CLC_ARG2_VECTYPE y, __CLC_ARG3_VECTYPE z) {
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0, y.s0, z.s0),
                            __CLC_IMPL_FUNCTION(x.s1, y.s1, z.s1),
                            __CLC_IMPL_FUNCTION(x.s2, y.s2, z.s2),
                            __CLC_IMPL_FUNCTION(x.s3, y.s3, z.s3),
                            __CLC_IMPL_FUNCTION(x.s4, y.s4, z.s4),
                            __CLC_IMPL_FUNCTION(x.s5, y.s5, z.s5),
                            __CLC_IMPL_FUNCTION(x.s6, y.s6, z.s6),
                            __CLC_IMPL_FUNCTION(x.s7, y.s7, z.s7),
                            __CLC_IMPL_FUNCTION(x.s8, y.s8, z.s8),
                            __CLC_IMPL_FUNCTION(x.s9, y.s9, z.s9),
                            __CLC_IMPL_FUNCTION(x.sa, y.sa, z.sa),
                            __CLC_IMPL_FUNCTION(x.sb, y.sb, z.sb),
                            __CLC_IMPL_FUNCTION(x.sc, y.sc, z.sc),
                            __CLC_IMPL_FUNCTION(x.sd, y.sd, z.sd),
                            __CLC_IMPL_FUNCTION(x.se, y.se, z.se),
                            __CLC_IMPL_FUNCTION(x.sf, y.sf, z.sf));
}
#undef __CLC_VECTOR_SIZE

#undef __CLC_RET_VECTYPE
#undef __CLC_ARG1_VECTYPE
#undef __CLC_ARG2_VECTYPE
#undef __CLC_ARG3_VECTYPE

#endif // __CLC_SCALAR
