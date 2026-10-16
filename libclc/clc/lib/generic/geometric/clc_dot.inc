//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if (__CLC_VECSIZE_OR_1 == 1 || __CLC_VECSIZE_OR_1 == 2 ||                     \
     __CLC_VECSIZE_OR_1 == 3 || __CLC_VECSIZE_OR_1 == 4)

_CLC_OVERLOAD _CLC_DEF __CLC_SCALAR_GENTYPE __clc_dot(__CLC_GENTYPE x,
                                                      __CLC_GENTYPE y) {
#if __CLC_VECSIZE_OR_1 == 1
  return x * y;
#elif __CLC_VECSIZE_OR_1 == 2
  return x.s0 * y.s0 + x.s1 * y.s1;
#elif __CLC_VECSIZE_OR_1 == 3
  return x.s0 * y.s0 + x.s1 * y.s1 + x.s2 * y.s2;
#else
  return x.s0 * y.s0 + x.s1 * y.s1 + x.s2 * y.s2 + x.s3 * y.s3;
#endif
}

#endif
