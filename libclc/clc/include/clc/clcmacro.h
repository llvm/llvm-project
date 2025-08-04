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

#define _CLC_V_V_VP_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG1_TYPE,         \
                              ADDR_SPACE, ARG2_TYPE)                           \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 2)                                          \
      FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 2) x,                                  \
               ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 2) * y) {                   \
    ADDR_SPACE ARG2_TYPE *ptr = (ADDR_SPACE ARG2_TYPE *)y;                     \
    return (__CLC_XCONCAT(RET_TYPE, 2))(FUNCTION(x.s0, ptr),                   \
                                        FUNCTION(x.s1, ptr + 1));              \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 3)                                          \
      FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 3) x,                                  \
               ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 3) * y) {                   \
    ADDR_SPACE ARG2_TYPE *ptr = (ADDR_SPACE ARG2_TYPE *)y;                     \
    return (__CLC_XCONCAT(RET_TYPE, 3))(FUNCTION(x.s0, ptr),                   \
                                        FUNCTION(x.s1, ptr + 1),               \
                                        FUNCTION(x.s2, ptr + 2));              \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 4)                                          \
      FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 4) x,                                  \
               ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 4) * y) {                   \
    ADDR_SPACE ARG2_TYPE *ptr = (ADDR_SPACE ARG2_TYPE *)y;                     \
    return (__CLC_XCONCAT(RET_TYPE, 4))(                                       \
        FUNCTION(x.s0, ptr), FUNCTION(x.s1, ptr + 1), FUNCTION(x.s2, ptr + 2), \
        FUNCTION(x.s3, ptr + 3));                                              \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 8)                                          \
      FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 8) x,                                  \
               ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 8) * y) {                   \
    ADDR_SPACE ARG2_TYPE *ptr = (ADDR_SPACE ARG2_TYPE *)y;                     \
    return (__CLC_XCONCAT(RET_TYPE, 8))(                                       \
        FUNCTION(x.s0, ptr), FUNCTION(x.s1, ptr + 1), FUNCTION(x.s2, ptr + 2), \
        FUNCTION(x.s3, ptr + 3), FUNCTION(x.s4, ptr + 4),                      \
        FUNCTION(x.s5, ptr + 5), FUNCTION(x.s6, ptr + 6),                      \
        FUNCTION(x.s7, ptr + 7));                                              \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 16)                                         \
      FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 16) x,                                 \
               ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 16) * y) {                  \
    ADDR_SPACE ARG2_TYPE *ptr = (ADDR_SPACE ARG2_TYPE *)y;                     \
    return (__CLC_XCONCAT(RET_TYPE, 16))(                                      \
        FUNCTION(x.s0, ptr), FUNCTION(x.s1, ptr + 1), FUNCTION(x.s2, ptr + 2), \
        FUNCTION(x.s3, ptr + 3), FUNCTION(x.s4, ptr + 4),                      \
        FUNCTION(x.s5, ptr + 5), FUNCTION(x.s6, ptr + 6),                      \
        FUNCTION(x.s7, ptr + 7), FUNCTION(x.s8, ptr + 8),                      \
        FUNCTION(x.s9, ptr + 9), FUNCTION(x.sa, ptr + 10),                     \
        FUNCTION(x.sb, ptr + 11), FUNCTION(x.sc, ptr + 12),                    \
        FUNCTION(x.sd, ptr + 13), FUNCTION(x.se, ptr + 14),                    \
        FUNCTION(x.sf, ptr + 15));                                             \
  }

#endif // __CLC_CLCMACRO_H__
