//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Do not generate user-facing default conversions for clspv as they are handled
// natively. Do not generate user-facing "_rte" conversions for clspv as they
// are handled natively.

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE_SATURATION_IMPL(TYPE_SRC, MODE)    \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION_SAT,       \
                                                     MODE)(TYPE_SRC x) {       \
    return __CLC_XCONCAT(__CLC_IMPL_FUNCTION_SAT, MODE)(x);                    \
  }

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE_SATURATION(MODE)                   \
  __CLC_DEFINE_FUNCTION_ROUNDING_MODE_SATURATION_IMPL(__CLC_S_GENTYPE_SRC,     \
                                                      MODE)                    \
  __CLC_DEFINE_FUNCTION_ROUNDING_MODE_SATURATION_IMPL(__CLC_U_GENTYPE_SRC, MODE)

__CLC_DEFINE_FUNCTION_ROUNDING_MODE_SATURATION()
__CLC_DEFINE_FUNCTION_ROUNDING_MODE_SATURATION(_rte)

#undef __CLC_DEFINE_FUNCTION_ROUNDING_MODE_SATURATION_IMPL
#undef __CLC_DEFINE_FUNCTION_ROUNDING_MODE_SATURATION

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE_IMPL(TYPE_SRC, MODE)               \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION,           \
                                                     MODE)(TYPE_SRC x) {       \
    return __CLC_XCONCAT(__CLC_IMPL_FUNCTION, MODE)(x);                        \
  }                                                                            \
  _CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __CLC_XCONCAT(__CLC_FUNCTION_SAT,       \
                                                     MODE)(TYPE_SRC x) {       \
    return __CLC_XCONCAT(__CLC_IMPL_FUNCTION_SAT, MODE)(x);                    \
  }

#define __CLC_DEFINE_FUNCTION_ROUNDING_MODE(MODE)                              \
  __CLC_DEFINE_FUNCTION_ROUNDING_MODE_IMPL(__CLC_S_GENTYPE_SRC, MODE)          \
  __CLC_DEFINE_FUNCTION_ROUNDING_MODE_IMPL(__CLC_U_GENTYPE_SRC, MODE)

__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtz)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtp)
__CLC_DEFINE_FUNCTION_ROUNDING_MODE(_rtn)

#undef __CLC_DEFINE_FUNCTION_ROUNDING_MODE_IMPL
#undef __CLC_DEFINE_FUNCTION_ROUNDING_MODE
