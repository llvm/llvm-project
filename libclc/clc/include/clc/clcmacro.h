//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CLC_CLCMACRO_H__
#define __CLC_CLCMACRO_H__

#include <clc/internal/clc.h>
#include <clc/utils.h>

#define _CLC_V_V_VP_VECTORIZE(DECLSPEC, RET_TYPE, __CLC_FUNCTION, ARG1_TYPE,   \
                              ADDR_SPACE, ARG2_TYPE)                           \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 2)                                          \
      __CLC_FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 2) x,                            \
                     ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 2) * y) {             \
    ADDR_SPACE ARG2_TYPE *ptr = (ADDR_SPACE ARG2_TYPE *)y;                     \
    return (__CLC_XCONCAT(RET_TYPE, 2))(__CLC_FUNCTION(x.s0, ptr),             \
                                        __CLC_FUNCTION(x.s1, ptr + 1));        \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 3)                                          \
      __CLC_FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 3) x,                            \
                     ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 3) * y) {             \
    ADDR_SPACE ARG2_TYPE *ptr = (ADDR_SPACE ARG2_TYPE *)y;                     \
    return (__CLC_XCONCAT(RET_TYPE, 3))(__CLC_FUNCTION(x.s0, ptr),             \
                                        __CLC_FUNCTION(x.s1, ptr + 1),         \
                                        __CLC_FUNCTION(x.s2, ptr + 2));        \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 4)                                          \
      __CLC_FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 4) x,                            \
                     ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 4) * y) {             \
    ADDR_SPACE ARG2_TYPE *ptr = (ADDR_SPACE ARG2_TYPE *)y;                     \
    return (__CLC_XCONCAT(RET_TYPE, 4))(                                       \
        __CLC_FUNCTION(x.s0, ptr), __CLC_FUNCTION(x.s1, ptr + 1),              \
        __CLC_FUNCTION(x.s2, ptr + 2), __CLC_FUNCTION(x.s3, ptr + 3));         \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 8)                                          \
      __CLC_FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 8) x,                            \
                     ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 8) * y) {             \
    ADDR_SPACE ARG2_TYPE *ptr = (ADDR_SPACE ARG2_TYPE *)y;                     \
    return (__CLC_XCONCAT(RET_TYPE, 8))(                                       \
        __CLC_FUNCTION(x.s0, ptr), __CLC_FUNCTION(x.s1, ptr + 1),              \
        __CLC_FUNCTION(x.s2, ptr + 2), __CLC_FUNCTION(x.s3, ptr + 3),          \
        __CLC_FUNCTION(x.s4, ptr + 4), __CLC_FUNCTION(x.s5, ptr + 5),          \
        __CLC_FUNCTION(x.s6, ptr + 6), __CLC_FUNCTION(x.s7, ptr + 7));         \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 16)                                         \
      __CLC_FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 16) x,                           \
                     ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 16) * y) {            \
    ADDR_SPACE ARG2_TYPE *ptr = (ADDR_SPACE ARG2_TYPE *)y;                     \
    return (__CLC_XCONCAT(RET_TYPE, 16))(                                      \
        __CLC_FUNCTION(x.s0, ptr), __CLC_FUNCTION(x.s1, ptr + 1),              \
        __CLC_FUNCTION(x.s2, ptr + 2), __CLC_FUNCTION(x.s3, ptr + 3),          \
        __CLC_FUNCTION(x.s4, ptr + 4), __CLC_FUNCTION(x.s5, ptr + 5),          \
        __CLC_FUNCTION(x.s6, ptr + 6), __CLC_FUNCTION(x.s7, ptr + 7),          \
        __CLC_FUNCTION(x.s8, ptr + 8), __CLC_FUNCTION(x.s9, ptr + 9),          \
        __CLC_FUNCTION(x.sa, ptr + 10), __CLC_FUNCTION(x.sb, ptr + 11),        \
        __CLC_FUNCTION(x.sc, ptr + 12), __CLC_FUNCTION(x.sd, ptr + 13),        \
        __CLC_FUNCTION(x.se, ptr + 14), __CLC_FUNCTION(x.sf, ptr + 15));       \
  }

#endif // __CLC_CLCMACRO_H__
