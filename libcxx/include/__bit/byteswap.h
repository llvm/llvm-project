// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___BIT_BYTESWAP_H
#define _LIBCPP___BIT_BYTESWAP_H

#include <__concepts/arithmetic.h>
#include <__config>
#include <__type_traits/is_same.h>
#include <__type_traits/remove_cv.h>
#include <climits>
#include <cstdint>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

template <integral _Tp>
[[nodiscard]] constexpr _Tp byteswap(_Tp __val) noexcept {
  // [bit.byteswap]/Mandates: T does not have padding bits.
  // bool is grandfathered: every shipping implementation admits it and the
  // size-1 identity path can't shuffle padding bits into value positions.
  // LWG 4583 proposes relaxing this to allow byte-aligned padding (e.g.
  // _BitInt(48) where 2 whole bytes are padding); revisit once it resolves.
  static_assert(is_same_v<remove_cv_t<_Tp>, bool> ||
                    numeric_limits<_Tp>::digits + numeric_limits<_Tp>::is_signed == sizeof(_Tp) * CHAR_BIT,
                "std::byteswap requires T to have no padding bits");

  if constexpr (sizeof(_Tp) == 1) {
    return __val;
#  if __has_builtin(__builtin_bswapg)
  } else {
    return __builtin_bswapg(__val);
  }
#  else
  } else if constexpr (sizeof(_Tp) == 2) {
    return __builtin_bswap16(__val);
  } else if constexpr (sizeof(_Tp) == 4) {
    return __builtin_bswap32(__val);
  } else if constexpr (sizeof(_Tp) == 8) {
    return __builtin_bswap64(__val);
#    if _LIBCPP_HAS_INT128
  } else if constexpr (sizeof(_Tp) == 16) {
#      if __has_builtin(__builtin_bswap128)
    return __builtin_bswap128(__val);
#      else
    return (static_cast<_Tp>(byteswap(static_cast<uint64_t>(__val))) << 64) |
           static_cast<_Tp>(byteswap(static_cast<uint64_t>(__val >> 64)));
#      endif // __has_builtin(__builtin_bswap128)
#    endif   // _LIBCPP_HAS_INT128
  } else {
    static_assert(sizeof(_Tp) == 0, "byteswap is unimplemented for integral types of this size");
  }
#  endif     // __has_builtin(__builtin_bswapg)
}

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___BIT_BYTESWAP_H
