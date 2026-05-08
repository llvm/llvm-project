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
#include <__cstddef/size_t.h>
#include <climits>
#include <cstdint>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

template <integral _Tp>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr _Tp byteswap(_Tp __val) noexcept {
  if constexpr (sizeof(_Tp) == 1) {
    // Identity for size-1 types: no bytes move and no padding gets shuffled
    // into significant positions. bool, char, and _BitInt(N <= CHAR_BIT)
    // all land here.
    return __val;
  } else {
    // Reject types whose value bits do not fill the entire object
    // representation (e.g. _BitInt(13) has 3 padding bits in 2 bytes of
    // storage). The byte-level builtins below would swap those padding
    // bits into significant positions, and the resulting value's meaning
    // is unspecified. The size-1 case above is exempt because no bytes
    // move.
    static_assert(numeric_limits<_Tp>::digits + numeric_limits<_Tp>::is_signed == sizeof(_Tp) * CHAR_BIT,
                  "std::byteswap requires a type whose value bits fill the entire "
                  "object representation; types like _BitInt(N) where N is not a "
                  "multiple of CHAR_BIT have padding bits and are rejected");
    if constexpr (sizeof(_Tp) == 2) {
      return __builtin_bswap16(__val);
    } else if constexpr (sizeof(_Tp) == 4) {
      return __builtin_bswap32(__val);
    } else if constexpr (sizeof(_Tp) == 8) {
      return __builtin_bswap64(__val);
#  if _LIBCPP_HAS_INT128
    } else if constexpr (sizeof(_Tp) == 16) {
#    if __has_builtin(__builtin_bswap128)
      return __builtin_bswap128(__val);
#    else
      return (static_cast<_Tp>(byteswap(static_cast<uint64_t>(__val))) << 64) |
             static_cast<_Tp>(byteswap(static_cast<uint64_t>(__val >> 64)));
#    endif // __has_builtin(__builtin_bswap128)
#  endif   // _LIBCPP_HAS_INT128
    } else {
      // Generic byte-reversal for wide integer types (e.g. _BitInt(N) with
      // N > 128). Reads the value 8 bits at a time and writes the bytes
      // back in reverse order. Left-shift on signed integral types is
      // well-defined modulo 2^width since C++20.
      _Tp __result = 0;
      for (size_t __i = 0; __i < sizeof(_Tp); ++__i) {
        __result |= static_cast<_Tp>(static_cast<unsigned char>(__val >> (__i * CHAR_BIT)))
                 << ((sizeof(_Tp) - 1 - __i) * CHAR_BIT);
      }
      return __result;
    }
  }
}

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___BIT_BYTESWAP_H
