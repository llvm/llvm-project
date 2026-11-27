//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifdef __CLC_F2F

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE(MODE)                              \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION, MODE)(    \
      __CLC_GENTYPE_SRC x) {                                                   \
    return __CLC_XCONCAT(__CLC_IMPL_FUNCTION, MODE)(x);                        \
  }

#elif defined(__CLC_I2F)

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE(MODE)                              \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION, MODE)(    \
      __CLC_S_GENTYPE_SRC x) {                                                 \
    return __CLC_XCONCAT(__CLC_IMPL_FUNCTION, MODE)(x);                        \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION, MODE)(    \
      __CLC_U_GENTYPE_SRC x) {                                                 \
    return __CLC_XCONCAT(__CLC_IMPL_FUNCTION, MODE)(x);                        \
  }

#elif defined(__CLC_F2I)

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE(MODE)                              \
  _CLC_OVERLOAD _CLC_DEF __CLC_S_GENTYPE_DST __CLC_XCONCAT(                    \
      __CLC_FUNCTION_S, MODE)(__CLC_GENTYPE x) {                               \
    return __CLC_XCONCAT(__CLC_IMPL_FUNCTION_S, MODE)(x);                      \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_S_GENTYPE_DST __CLC_XCONCAT(                    \
      __CLC_FUNCTION_S_SAT, MODE)(__CLC_GENTYPE x) {                           \
    return __CLC_XCONCAT(__CLC_IMPL_FUNCTION_S_SAT, MODE)(x);                  \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE_DST __CLC_XCONCAT(                    \
      __CLC_FUNCTION_U, MODE)(__CLC_GENTYPE x) {                               \
    return __CLC_XCONCAT(__CLC_IMPL_FUNCTION_U, MODE)(x);                      \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_U_GENTYPE_DST __CLC_XCONCAT(                    \
      __CLC_FUNCTION_U_SAT, MODE)(__CLC_GENTYPE x) {                           \
    return __CLC_XCONCAT(__CLC_IMPL_FUNCTION_U_SAT, MODE)(x);                  \
  }

#endif

__CLC_DEFINE_FUNCTION_ROUNDING_MODE()
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rte)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtz)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtp)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtn)

#undef __CLC_DEFINE_FUNCTION_ROUNDING_MODE
