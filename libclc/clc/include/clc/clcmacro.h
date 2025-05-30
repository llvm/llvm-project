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

#define _CLC_UNARY_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG1_TYPE)          \
  DECLSPEC RET_TYPE##2 FUNCTION(ARG1_TYPE##2 x) {                              \
    return (RET_TYPE##2)(FUNCTION(x.s0), FUNCTION(x.s1));                      \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##3 FUNCTION(ARG1_TYPE##3 x) {                              \
    return (RET_TYPE##3)(FUNCTION(x.s0), FUNCTION(x.s1), FUNCTION(x.s2));      \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##4 FUNCTION(ARG1_TYPE##4 x) {                              \
    return (RET_TYPE##4)(FUNCTION(x.s0), FUNCTION(x.s1), FUNCTION(x.s2),       \
                         FUNCTION(x.s3));                                      \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##8 FUNCTION(ARG1_TYPE##8 x) {                              \
    return (RET_TYPE##8)(FUNCTION(x.s0), FUNCTION(x.s1), FUNCTION(x.s2),       \
                         FUNCTION(x.s3), FUNCTION(x.s4), FUNCTION(x.s5),       \
                         FUNCTION(x.s6), FUNCTION(x.s7));                      \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##16 FUNCTION(ARG1_TYPE##16 x) {                            \
    return (RET_TYPE##16)(                                                     \
        FUNCTION(x.s0), FUNCTION(x.s1), FUNCTION(x.s2), FUNCTION(x.s3),        \
        FUNCTION(x.s4), FUNCTION(x.s5), FUNCTION(x.s6), FUNCTION(x.s7),        \
        FUNCTION(x.s8), FUNCTION(x.s9), FUNCTION(x.sa), FUNCTION(x.sb),        \
        FUNCTION(x.sc), FUNCTION(x.sd), FUNCTION(x.se), FUNCTION(x.sf));       \
  }

#define _CLC_BINARY_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG1_TYPE,         \
                              ARG2_TYPE)                                       \
  DECLSPEC RET_TYPE##2 FUNCTION(ARG1_TYPE##2 x, ARG2_TYPE##2 y) {              \
    return (RET_TYPE##2)(FUNCTION(x.s0, y.s0), FUNCTION(x.s1, y.s1));          \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##3 FUNCTION(ARG1_TYPE##3 x, ARG2_TYPE##3 y) {              \
    return (RET_TYPE##3)(FUNCTION(x.s0, y.s0), FUNCTION(x.s1, y.s1),           \
                         FUNCTION(x.s2, y.s2));                                \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##4 FUNCTION(ARG1_TYPE##4 x, ARG2_TYPE##4 y) {              \
    return (RET_TYPE##4)(FUNCTION(x.s0, y.s0), FUNCTION(x.s1, y.s1),           \
                         FUNCTION(x.s2, y.s2), FUNCTION(x.s3, y.s3));          \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##8 FUNCTION(ARG1_TYPE##8 x, ARG2_TYPE##8 y) {              \
    return (RET_TYPE##8)(FUNCTION(x.s0, y.s0), FUNCTION(x.s1, y.s1),           \
                         FUNCTION(x.s2, y.s2), FUNCTION(x.s3, y.s3),           \
                         FUNCTION(x.s4, y.s4), FUNCTION(x.s5, y.s5),           \
                         FUNCTION(x.s6, y.s6), FUNCTION(x.s7, y.s7));          \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##16 FUNCTION(ARG1_TYPE##16 x, ARG2_TYPE##16 y) {           \
    return (RET_TYPE##16)(                                                     \
        FUNCTION(x.s0, y.s0), FUNCTION(x.s1, y.s1), FUNCTION(x.s2, y.s2),      \
        FUNCTION(x.s3, y.s3), FUNCTION(x.s4, y.s4), FUNCTION(x.s5, y.s5),      \
        FUNCTION(x.s6, y.s6), FUNCTION(x.s7, y.s7), FUNCTION(x.s8, y.s8),      \
        FUNCTION(x.s9, y.s9), FUNCTION(x.sa, y.sa), FUNCTION(x.sb, y.sb),      \
        FUNCTION(x.sc, y.sc), FUNCTION(x.sd, y.sd), FUNCTION(x.se, y.se),      \
        FUNCTION(x.sf, y.sf));                                                 \
  }

#define _CLC_TERNARY_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG1_TYPE,        \
                               ARG2_TYPE, ARG3_TYPE)                           \
  DECLSPEC RET_TYPE##2 FUNCTION(ARG1_TYPE##2 x, ARG2_TYPE##2 y,                \
                                ARG3_TYPE##2 z) {                              \
    return (RET_TYPE##2)(FUNCTION(x.s0, y.s0, z.s0),                           \
                         FUNCTION(x.s1, y.s1, z.s1));                          \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##3 FUNCTION(ARG1_TYPE##3 x, ARG2_TYPE##3 y,                \
                                ARG3_TYPE##3 z) {                              \
    return (RET_TYPE##3)(FUNCTION(x.s0, y.s0, z.s0),                           \
                         FUNCTION(x.s1, y.s1, z.s1),                           \
                         FUNCTION(x.s2, y.s2, z.s2));                          \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##4 FUNCTION(ARG1_TYPE##4 x, ARG2_TYPE##4 y,                \
                                ARG3_TYPE##4 z) {                              \
    return (RET_TYPE##4)(                                                      \
        FUNCTION(x.s0, y.s0, z.s0), FUNCTION(x.s1, y.s1, z.s1),                \
        FUNCTION(x.s2, y.s2, z.s2), FUNCTION(x.s3, y.s3, z.s3));               \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##8 FUNCTION(ARG1_TYPE##8 x, ARG2_TYPE##8 y,                \
                                ARG3_TYPE##8 z) {                              \
    return (RET_TYPE##8)(                                                      \
        FUNCTION(x.s0, y.s0, z.s0), FUNCTION(x.s1, y.s1, z.s1),                \
        FUNCTION(x.s2, y.s2, z.s2), FUNCTION(x.s3, y.s3, z.s3),                \
        FUNCTION(x.s4, y.s4, z.s4), FUNCTION(x.s5, y.s5, z.s5),                \
        FUNCTION(x.s6, y.s6, z.s6), FUNCTION(x.s7, y.s7, z.s7));               \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##16 FUNCTION(ARG1_TYPE##16 x, ARG2_TYPE##16 y,             \
                                 ARG3_TYPE##16 z) {                            \
    return (RET_TYPE##16)(                                                     \
        FUNCTION(x.s0, y.s0, z.s0), FUNCTION(x.s1, y.s1, z.s1),                \
        FUNCTION(x.s2, y.s2, z.s2), FUNCTION(x.s3, y.s3, z.s3),                \
        FUNCTION(x.s4, y.s4, z.s4), FUNCTION(x.s5, y.s5, z.s5),                \
        FUNCTION(x.s6, y.s6, z.s6), FUNCTION(x.s7, y.s7, z.s7),                \
        FUNCTION(x.s8, y.s8, z.s8), FUNCTION(x.s9, y.s9, z.s9),                \
        FUNCTION(x.sa, y.sa, z.sa), FUNCTION(x.sb, y.sb, z.sb),                \
        FUNCTION(x.sc, y.sc, z.sc), FUNCTION(x.sd, y.sd, z.sd),                \
        FUNCTION(x.se, y.se, z.se), FUNCTION(x.sf, y.sf, z.sf));               \
  }

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

#define _CLC_DEFINE_BINARY_BUILTIN(RET_TYPE, FUNCTION, BUILTIN, ARG1_TYPE,     \
                                   ARG2_TYPE)                                  \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG1_TYPE x, ARG2_TYPE y) {         \
    return BUILTIN(x, y);                                                      \
  }                                                                            \
  _CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, RET_TYPE, FUNCTION, ARG1_TYPE, \
                        ARG2_TYPE)

// FIXME: Make _CLC_DEFINE_BINARY_BUILTIN avoid scalarization by default, and
// introduce an explicit scalarizing version.
#define _CLC_DEFINE_BINARY_BUILTIN_NO_SCALARIZE(RET_TYPE, FUNCTION, BUILTIN,   \
                                                ARG1_TYPE, ARG2_TYPE)          \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG1_TYPE x, ARG2_TYPE y) {         \
    return BUILTIN(x, y);                                                      \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##2 FUNCTION(ARG1_TYPE##2 x,                  \
                                              ARG2_TYPE##2 y) {                \
    return BUILTIN(x, y);                                                      \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##3 FUNCTION(ARG1_TYPE##3 x,                  \
                                              ARG2_TYPE##3 y) {                \
    return BUILTIN(x, y);                                                      \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##4 FUNCTION(ARG1_TYPE##4 x,                  \
                                              ARG2_TYPE##4 y) {                \
    return BUILTIN(x, y);                                                      \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##8 FUNCTION(ARG1_TYPE##8 x,                  \
                                              ARG2_TYPE##8 y) {                \
    return BUILTIN(x, y);                                                      \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##16 FUNCTION(ARG1_TYPE##16 x,                \
                                               ARG2_TYPE##16 y) {              \
    return BUILTIN(x, y);                                                      \
  }

#define _CLC_DEFINE_BINARY_BUILTIN_WITH_SCALAR_SECOND_ARG(                     \
    RET_TYPE, FUNCTION, BUILTIN, ARG1_TYPE, ARG2_TYPE)                         \
  _CLC_DEFINE_BINARY_BUILTIN(RET_TYPE, FUNCTION, BUILTIN, ARG1_TYPE,           \
                             ARG2_TYPE)                                        \
  _CLC_BINARY_VECTORIZE_SCALAR_SECOND_ARG(_CLC_OVERLOAD _CLC_DEF, RET_TYPE,    \
                                          FUNCTION, ARG1_TYPE, ARG2_TYPE)

#define _CLC_DEFINE_UNARY_BUILTIN(RET_TYPE, FUNCTION, BUILTIN, ARG1_TYPE)      \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG1_TYPE x) { return BUILTIN(x); } \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##2 FUNCTION(ARG1_TYPE##2 x) {                \
    return BUILTIN(x);                                                         \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##3 FUNCTION(ARG1_TYPE##3 x) {                \
    return BUILTIN(x);                                                         \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##4 FUNCTION(ARG1_TYPE##4 x) {                \
    return BUILTIN(x);                                                         \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##8 FUNCTION(ARG1_TYPE##8 x) {                \
    return BUILTIN(x);                                                         \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##16 FUNCTION(ARG1_TYPE##16 x) {              \
    return BUILTIN(x);                                                         \
  }

#define _CLC_DEFINE_TERNARY_BUILTIN(RET_TYPE, FUNCTION, BUILTIN, ARG1_TYPE,    \
                                    ARG2_TYPE, ARG3_TYPE)                      \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE FUNCTION(ARG1_TYPE x, ARG2_TYPE y,           \
                                           ARG3_TYPE z) {                      \
    return BUILTIN(x, y, z);                                                   \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##2 FUNCTION(ARG1_TYPE##2 x, ARG2_TYPE##2 y,  \
                                              ARG3_TYPE##2 z) {                \
    return BUILTIN(x, y, z);                                                   \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##3 FUNCTION(ARG1_TYPE##3 x, ARG2_TYPE##3 y,  \
                                              ARG3_TYPE##3 z) {                \
    return BUILTIN(x, y, z);                                                   \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##4 FUNCTION(ARG1_TYPE##4 x, ARG2_TYPE##4 y,  \
                                              ARG3_TYPE##4 z) {                \
    return BUILTIN(x, y, z);                                                   \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##8 FUNCTION(ARG1_TYPE##8 x, ARG2_TYPE##8 y,  \
                                              ARG3_TYPE##8 z) {                \
    return BUILTIN(x, y, z);                                                   \
  }                                                                            \
  _CLC_DEF _CLC_OVERLOAD RET_TYPE##16 FUNCTION(                                \
      ARG1_TYPE##16 x, ARG2_TYPE##16 y, ARG3_TYPE##16 z) {                     \
    return BUILTIN(x, y, z);                                                   \
  }

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define _CLC_DEFINE_UNARY_BUILTIN_FP16(FUNCTION)                               \
  _CLC_DEF _CLC_OVERLOAD half FUNCTION(half x) {                               \
    return (half)FUNCTION((float)x);                                           \
  }                                                                            \
  _CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, FUNCTION, half)

#define _CLC_DEFINE_BINARY_BUILTIN_FP16(FUNCTION)                              \
  _CLC_DEF _CLC_OVERLOAD half FUNCTION(half x, half y) {                       \
    return (half)FUNCTION((float)x, (float)y);                                 \
  }                                                                            \
  _CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, FUNCTION, half, half)

#pragma OPENCL EXTENSION cl_khr_fp16 : disable

#else

#define _CLC_DEFINE_UNARY_BUILTIN_FP16(FUNCTION)
#define _CLC_DEFINE_BINARY_BUILTIN_FP16(FUNCTION)

#endif

#endif // __CLC_CLCMACRO_H__
