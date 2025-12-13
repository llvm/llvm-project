//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/utils.h>

#ifdef __CLC_SCALAR

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
#define __CLC_ARG2_VECTYPE __CLC_XCONCAT(__CLC_ARG2_TYPE, __CLC_VECTOR_SIZE)

#define __CLC_VECTOR_SIZE 2
_CLC_OVERLOAD _CLC_DEF __CLC_RET_VECTYPE
__CLC_FUNCTION(__CLC_ARG1_VECTYPE x, __CLC_ADDRSPACE __CLC_ARG2_VECTYPE *ptr) {
  __CLC_ADDRSPACE __CLC_ARG2_TYPE *p = (__CLC_ADDRSPACE __CLC_ARG2_TYPE *)ptr;

  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0, p),
                             __CLC_IMPL_FUNCTION(x.s1, p + 1));
}
#undef __CLC_VECTOR_SIZE

#define __CLC_VECTOR_SIZE 3
_CLC_OVERLOAD _CLC_DEF __CLC_RET_VECTYPE
__CLC_FUNCTION(__CLC_ARG1_VECTYPE x, __CLC_ADDRSPACE __CLC_ARG2_VECTYPE *ptr) {
  __CLC_ADDRSPACE __CLC_ARG2_TYPE *p = (__CLC_ADDRSPACE __CLC_ARG2_TYPE *)ptr;
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0, p),
                             __CLC_IMPL_FUNCTION(x.s1, p + 1),
                             __CLC_IMPL_FUNCTION(x.s2, p + 2));
}
#undef __CLC_VECTOR_SIZE

#define __CLC_VECTOR_SIZE 4
_CLC_OVERLOAD _CLC_DEF __CLC_RET_VECTYPE
__CLC_FUNCTION(__CLC_ARG1_VECTYPE x, __CLC_ADDRSPACE __CLC_ARG2_VECTYPE *ptr) {
  __CLC_ADDRSPACE __CLC_ARG2_TYPE *p = (__CLC_ADDRSPACE __CLC_ARG2_TYPE *)ptr;
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0, p),
                             __CLC_IMPL_FUNCTION(x.s1, p + 1),
                             __CLC_IMPL_FUNCTION(x.s2, p + 2),
                             __CLC_IMPL_FUNCTION(x.s3, p + 3));
}
#undef __CLC_VECTOR_SIZE

#define __CLC_VECTOR_SIZE 8
_CLC_OVERLOAD _CLC_DEF __CLC_RET_VECTYPE
__CLC_FUNCTION(__CLC_ARG1_VECTYPE x, __CLC_ADDRSPACE __CLC_ARG2_VECTYPE *ptr) {
  __CLC_ADDRSPACE __CLC_ARG2_TYPE *p = (__CLC_ADDRSPACE __CLC_ARG2_TYPE *)ptr;
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0, p),
                             __CLC_IMPL_FUNCTION(x.s1, p + 1),
                             __CLC_IMPL_FUNCTION(x.s2, p + 2),
                             __CLC_IMPL_FUNCTION(x.s3, p + 3),
                             __CLC_IMPL_FUNCTION(x.s4, p + 4),
                             __CLC_IMPL_FUNCTION(x.s5, p + 5),
                             __CLC_IMPL_FUNCTION(x.s6, p + 6),
                             __CLC_IMPL_FUNCTION(x.s7, p + 7));
}
#undef __CLC_VECTOR_SIZE

#define __CLC_VECTOR_SIZE 16
_CLC_OVERLOAD _CLC_DEF __CLC_RET_VECTYPE
__CLC_FUNCTION(__CLC_ARG1_VECTYPE x, __CLC_ADDRSPACE __CLC_ARG2_VECTYPE *ptr) {
  __CLC_ADDRSPACE __CLC_ARG2_TYPE *p = (__CLC_ADDRSPACE __CLC_ARG2_TYPE *)ptr;
  return (__CLC_RET_VECTYPE)(__CLC_IMPL_FUNCTION(x.s0, p),
                             __CLC_IMPL_FUNCTION(x.s1, p + 1),
                             __CLC_IMPL_FUNCTION(x.s2, p + 2),
                             __CLC_IMPL_FUNCTION(x.s3, p + 3),
                             __CLC_IMPL_FUNCTION(x.s4, p + 4),
                             __CLC_IMPL_FUNCTION(x.s5, p + 5),
                             __CLC_IMPL_FUNCTION(x.s6, p + 6),
                             __CLC_IMPL_FUNCTION(x.s7, p + 7),
                             __CLC_IMPL_FUNCTION(x.s8, p + 8),
                             __CLC_IMPL_FUNCTION(x.s9, p + 9),
                             __CLC_IMPL_FUNCTION(x.sa, p + 10),
                             __CLC_IMPL_FUNCTION(x.sb, p + 11),
                             __CLC_IMPL_FUNCTION(x.sc, p + 12),
                             __CLC_IMPL_FUNCTION(x.sd, p + 13),
                             __CLC_IMPL_FUNCTION(x.se, p + 14),
                             __CLC_IMPL_FUNCTION(x.sf, p + 15));
}
#undef __CLC_VECTOR_SIZE

#undef __CLC_RET_VECTYPE
#undef __CLC_ARG1_VECTYPE
#undef __CLC_ARG2_VECTYPE

#endif // __CLC_SCALAR
