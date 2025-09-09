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

#ifndef __CLC_RET_TYPE
#define __CLC_RET_TYPE __CLC_GENTYPE
#endif

#ifndef __CLC_ARG1_TYPE
#define __CLC_ARG1_TYPE __CLC_GENTYPE
#endif

#ifndef __CLC_ARG2_TYPE
#define __CLC_ARG2_TYPE __CLC_GENTYPE
#endif

#define __CLC_RET_VECTYPE __CLC_XCONCAT(__CLC_RET_TYPE, __CLC_VECTOR_SIZE)
#define __CLC_ARG1_VECTYPE __CLC_XCONCAT(__CLC_ARG1_TYPE, __CLC_VECTOR_SIZE)

#if __CLC_MIN_VECSIZE == 1
_CLC_OVERLOAD _CLC_DEF __CLC_RET_TYPE __CLC_FUNCTION(__CLC_ARG1_TYPE x) {
  return __CLC_IMPL_FUNCTION(x);
}
#endif // __CLC_MIN_VECSIZE == 1

#if __CLC_MIN_VECSIZE <= 2
#define __CLC_VECTOR_SIZE 2
_CLC_OVERLOAD _CLC_DEF __CLC_RET_VECTYPE __CLC_FUNCTION(__CLC_ARG1_VECTYPE x) {
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0),
                           __CLC_IMPL_FUNCTION(x.s1));
}
#undef __CLC_VECTOR_SIZE
#endif // __CLC_MIN_VECSIZE <= 2

#if __CLC_MIN_VECSIZE > 3
#error "__CLC_MIN_VECSIZE > 3 isn't implemented"
#endif

#define __CLC_VECTOR_SIZE 3
_CLC_OVERLOAD _CLC_DEF __CLC_RET_VECTYPE __CLC_FUNCTION(__CLC_ARG1_VECTYPE x) {
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0), __CLC_IMPL_FUNCTION(x.s1),
                           __CLC_IMPL_FUNCTION(x.s2));
}
#undef __CLC_VECTOR_SIZE

#define __CLC_VECTOR_SIZE 4
_CLC_OVERLOAD _CLC_DEF __CLC_RET_VECTYPE __CLC_FUNCTION(__CLC_ARG1_VECTYPE x) {
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0), __CLC_IMPL_FUNCTION(x.s1),
                           __CLC_IMPL_FUNCTION(x.s2),
                           __CLC_IMPL_FUNCTION(x.s3));
}
#undef __CLC_VECTOR_SIZE

#define __CLC_VECTOR_SIZE 8
_CLC_OVERLOAD _CLC_DEF __CLC_RET_VECTYPE __CLC_FUNCTION(__CLC_ARG1_VECTYPE x) {
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0), __CLC_IMPL_FUNCTION(x.s1),
                           __CLC_IMPL_FUNCTION(x.s2), __CLC_IMPL_FUNCTION(x.s3),
                           __CLC_IMPL_FUNCTION(x.s4), __CLC_IMPL_FUNCTION(x.s5),
                           __CLC_IMPL_FUNCTION(x.s6),
                           __CLC_IMPL_FUNCTION(x.s7));
}
#undef __CLC_VECTOR_SIZE

#define __CLC_VECTOR_SIZE 16
_CLC_OVERLOAD _CLC_DEF __CLC_RET_VECTYPE __CLC_FUNCTION(__CLC_ARG1_VECTYPE x) {
  return (
      __CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0), __CLC_IMPL_FUNCTION(x.s1),
                        __CLC_IMPL_FUNCTION(x.s2), __CLC_IMPL_FUNCTION(x.s3),
                        __CLC_IMPL_FUNCTION(x.s4), __CLC_IMPL_FUNCTION(x.s5),
                        __CLC_IMPL_FUNCTION(x.s6), __CLC_IMPL_FUNCTION(x.s7),
                        __CLC_IMPL_FUNCTION(x.s8), __CLC_IMPL_FUNCTION(x.s9),
                        __CLC_IMPL_FUNCTION(x.sa), __CLC_IMPL_FUNCTION(x.sb),
                        __CLC_IMPL_FUNCTION(x.sc), __CLC_IMPL_FUNCTION(x.sd),
                        __CLC_IMPL_FUNCTION(x.se), __CLC_IMPL_FUNCTION(x.sf));
}
#undef __CLC_VECTOR_SIZE

#undef __CLC_RET_VECTYPE
#undef __CLC_ARG1_VECTYPE

#endif // __CLC_SCALAR
