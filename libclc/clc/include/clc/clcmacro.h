#ifndef __CLC_CLCMACRO_H__
#define __CLC_CLCMACRO_H__

#include <clc/internal/clc.h>
#include <clc/utils.h>

#define _CLC_UNARY_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG1_TYPE)          \
  DECLSPEC RET_TYPE##2 FUNCTION(ARG1_TYPE##2 x) {                              \
    return (RET_TYPE##2)(FUNCTION(x.x), FUNCTION(x.y));                        \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##3 FUNCTION(ARG1_TYPE##3 x) {                              \
    return (RET_TYPE##3)(FUNCTION(x.x), FUNCTION(x.y), FUNCTION(x.z));         \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##4 FUNCTION(ARG1_TYPE##4 x) {                              \
    return (RET_TYPE##4)(FUNCTION(x.lo), FUNCTION(x.hi));                      \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##8 FUNCTION(ARG1_TYPE##8 x) {                              \
    return (RET_TYPE##8)(FUNCTION(x.lo), FUNCTION(x.hi));                      \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##16 FUNCTION(ARG1_TYPE##16 x) {                            \
    return (RET_TYPE##16)(FUNCTION(x.lo), FUNCTION(x.hi));                     \
  }

#define _CLC_BINARY_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG1_TYPE,         \
                              ARG2_TYPE)                                       \
  DECLSPEC RET_TYPE##2 FUNCTION(ARG1_TYPE##2 x, ARG2_TYPE##2 y) {              \
    return (RET_TYPE##2)(FUNCTION(x.x, y.x), FUNCTION(x.y, y.y));              \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##3 FUNCTION(ARG1_TYPE##3 x, ARG2_TYPE##3 y) {              \
    return (RET_TYPE##3)(FUNCTION(x.x, y.x), FUNCTION(x.y, y.y),               \
                         FUNCTION(x.z, y.z));                                  \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##4 FUNCTION(ARG1_TYPE##4 x, ARG2_TYPE##4 y) {              \
    return (RET_TYPE##4)(FUNCTION(x.lo, y.lo), FUNCTION(x.hi, y.hi));          \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##8 FUNCTION(ARG1_TYPE##8 x, ARG2_TYPE##8 y) {              \
    return (RET_TYPE##8)(FUNCTION(x.lo, y.lo), FUNCTION(x.hi, y.hi));          \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##16 FUNCTION(ARG1_TYPE##16 x, ARG2_TYPE##16 y) {           \
    return (RET_TYPE##16)(FUNCTION(x.lo, y.lo), FUNCTION(x.hi, y.hi));         \
  }

#define _CLC_V_S_V_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG1_TYPE,          \
                             ARG2_TYPE)                                        \
  DECLSPEC RET_TYPE##2 FUNCTION(ARG1_TYPE x, ARG2_TYPE##2 y) {                 \
    return (RET_TYPE##2)(FUNCTION(x, y.lo), FUNCTION(x, y.hi));                \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##3 FUNCTION(ARG1_TYPE x, ARG2_TYPE##3 y) {                 \
    return (RET_TYPE##3)(FUNCTION(x, y.x), FUNCTION(x, y.y),                   \
                         FUNCTION(x, y.z));                                    \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##4 FUNCTION(ARG1_TYPE x, ARG2_TYPE##4 y) {                 \
    return (RET_TYPE##4)(FUNCTION(x, y.lo), FUNCTION(x, y.hi));                \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##8 FUNCTION(ARG1_TYPE x, ARG2_TYPE##8 y) {                 \
    return (RET_TYPE##8)(FUNCTION(x, y.lo), FUNCTION(x, y.hi));                \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##16 FUNCTION(ARG1_TYPE x, ARG2_TYPE##16 y) {               \
    return (RET_TYPE##16)(FUNCTION(x, y.lo), FUNCTION(x, y.hi));               \
  }

#define _CLC_TERNARY_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG1_TYPE,        \
                               ARG2_TYPE, ARG3_TYPE)                           \
  DECLSPEC RET_TYPE##2 FUNCTION(ARG1_TYPE##2 x, ARG2_TYPE##2 y,                \
                                ARG3_TYPE##2 z) {                              \
    return (RET_TYPE##2)(FUNCTION(x.x, y.x, z.x), FUNCTION(x.y, y.y, z.y));    \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##3 FUNCTION(ARG1_TYPE##3 x, ARG2_TYPE##3 y,                \
                                ARG3_TYPE##3 z) {                              \
    return (RET_TYPE##3)(FUNCTION(x.x, y.x, z.x), FUNCTION(x.y, y.y, z.y),     \
                         FUNCTION(x.z, y.z, z.z));                             \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##4 FUNCTION(ARG1_TYPE##4 x, ARG2_TYPE##4 y,                \
                                ARG3_TYPE##4 z) {                              \
    return (RET_TYPE##4)(FUNCTION(x.lo, y.lo, z.lo),                           \
                         FUNCTION(x.hi, y.hi, z.hi));                          \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##8 FUNCTION(ARG1_TYPE##8 x, ARG2_TYPE##8 y,                \
                                ARG3_TYPE##8 z) {                              \
    return (RET_TYPE##8)(FUNCTION(x.lo, y.lo, z.lo),                           \
                         FUNCTION(x.hi, y.hi, z.hi));                          \
  }                                                                            \
                                                                               \
  DECLSPEC RET_TYPE##16 FUNCTION(ARG1_TYPE##16 x, ARG2_TYPE##16 y,             \
                                 ARG3_TYPE##16 z) {                            \
    return (RET_TYPE##16)(FUNCTION(x.lo, y.lo, z.lo),                          \
                          FUNCTION(x.hi, y.hi, z.hi));                         \
  }

#define _CLC_V_V_VP_VECTORIZE(DECLSPEC, RET_TYPE, FUNCTION, ARG1_TYPE,         \
                              ADDR_SPACE, ARG2_TYPE)                           \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 2)                                          \
      FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 2) x,                                  \
               ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 2) * y) {                   \
    return (__CLC_XCONCAT(RET_TYPE, 2))(                                       \
        FUNCTION(x.x, (ADDR_SPACE ARG2_TYPE *)y),                              \
        FUNCTION(x.y,                                                          \
                 (ADDR_SPACE ARG2_TYPE *)((ADDR_SPACE ARG2_TYPE *)y + 1)));    \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 3)                                          \
      FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 3) x,                                  \
               ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 3) * y) {                   \
    return (__CLC_XCONCAT(RET_TYPE, 3))(                                       \
        FUNCTION(x.x, (ADDR_SPACE ARG2_TYPE *)y),                              \
        FUNCTION(x.y,                                                          \
                 (ADDR_SPACE ARG2_TYPE *)((ADDR_SPACE ARG2_TYPE *)y + 1)),     \
        FUNCTION(x.z,                                                          \
                 (ADDR_SPACE ARG2_TYPE *)((ADDR_SPACE ARG2_TYPE *)y + 2)));    \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 4)                                          \
      FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 4) x,                                  \
               ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 4) * y) {                   \
    return (__CLC_XCONCAT(RET_TYPE, 4))(                                       \
        FUNCTION(x.lo, (ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 2) *)y),           \
        FUNCTION(x.hi, (ADDR_SPACE __CLC_XCONCAT(                              \
                           ARG2_TYPE, 2) *)((ADDR_SPACE ARG2_TYPE *)y + 2)));  \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 8)                                          \
      FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 8) x,                                  \
               ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 8) * y) {                   \
    return (__CLC_XCONCAT(RET_TYPE, 8))(                                       \
        FUNCTION(x.lo, (ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 4) *)y),           \
        FUNCTION(x.hi, (ADDR_SPACE __CLC_XCONCAT(                              \
                           ARG2_TYPE, 4) *)((ADDR_SPACE ARG2_TYPE *)y + 4)));  \
  }                                                                            \
                                                                               \
  DECLSPEC __CLC_XCONCAT(RET_TYPE, 16)                                         \
      FUNCTION(__CLC_XCONCAT(ARG1_TYPE, 16) x,                                 \
               ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 16) * y) {                  \
    return (__CLC_XCONCAT(RET_TYPE, 16))(                                      \
        FUNCTION(x.lo, (ADDR_SPACE __CLC_XCONCAT(ARG2_TYPE, 8) *)y),           \
        FUNCTION(x.hi, (ADDR_SPACE __CLC_XCONCAT(                              \
                           ARG2_TYPE, 8) *)((ADDR_SPACE ARG2_TYPE *)y + 8)));  \
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
