// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_EXPERIMENTAL___SIMD_UTILITY_H
#define _LIBCPP_EXPERIMENTAL___SIMD_UTILITY_H

#include <__type_traits/is_arithmetic.h>
#include <__type_traits/is_const.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_volatile.h>
#include <cstdint>
#include <limits>

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

_LIBCPP_BEGIN_NAMESPACE_EXPERIMENTAL
inline namespace parallelism_v2 {
template <class _Tp>
constexpr bool __is_vectorizable_v =
    is_arithmetic_v<_Tp> && !is_const_v<_Tp> && !is_volatile_v<_Tp> && !is_same_v<_Tp, bool>;

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI auto __choose_mask_type() {
  if constexpr (sizeof(_Tp) == 1) {
    return uint8_t{};
  } else if constexpr (sizeof(_Tp) == 2) {
    return uint16_t{};
  } else if constexpr (sizeof(_Tp) == 4) {
    return uint32_t{};
  } else if constexpr (sizeof(_Tp) == 8) {
    return uint64_t{};
  }
#  ifndef _LIBCPP_HAS_NO_INT128
  else if constexpr (sizeof(_Tp) == 16) {
    return __uint128_t{};
  }
#  endif
  else
    static_assert(sizeof(_Tp) == 0, "Unexpected size");
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI auto constexpr __set_all_bits(bool __v) {
  return __v ? (numeric_limits<decltype(__choose_mask_type<_Tp>())>::max()) : 0;
}

} // namespace parallelism_v2
_LIBCPP_END_NAMESPACE_EXPERIMENTAL

#endif // _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)

_LIBCPP_POP_MACROS

#endif // _LIBCPP_EXPERIMENTAL___SIMD_UTILITY_H
