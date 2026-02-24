// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHARCONV_TO_CHARS_BASE_10_H
#define _LIBCPP___CHARCONV_TO_CHARS_BASE_10_H

#include <__algorithm/copy_n.h>
#include <__assert>
#include <__charconv/tables.h>
#include <__config>
#include <cstdint>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __itoa {

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char* __append1(char* __first, uint32_t __value) _NOEXCEPT {
  *__first = '0' + static_cast<char>(__value);
  return __first + 1;
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char* __append2(char* __first, uint32_t __value) _NOEXCEPT {
  return std::copy_n(&__digits_base_10[__value * 2], 2, __first);
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char* __append3(char* __first, uint32_t __value) _NOEXCEPT {
  return __itoa::__append2(__itoa::__append1(__first, __value / 100), __value % 100);
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char* __append4(char* __first, uint32_t __value) _NOEXCEPT {
  return __itoa::__append2(__itoa::__append2(__first, __value / 100), __value % 100);
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char* __append5(char* __first, uint32_t __value) _NOEXCEPT {
  return __itoa::__append4(__itoa::__append1(__first, __value / 10000), __value % 10000);
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char* __append6(char* __first, uint32_t __value) _NOEXCEPT {
  return __itoa::__append4(__itoa::__append2(__first, __value / 10000), __value % 10000);
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char* __append7(char* __first, uint32_t __value) _NOEXCEPT {
  return __itoa::__append6(__itoa::__append1(__first, __value / 1000000), __value % 1000000);
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char* __append8(char* __first, uint32_t __value) _NOEXCEPT {
  return __itoa::__append6(__itoa::__append2(__first, __value / 1000000), __value % 1000000);
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char* __append9(char* __first, uint32_t __value) _NOEXCEPT {
  return __itoa::__append8(__itoa::__append1(__first, __value / 100000000), __value % 100000000);
}

template <class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI char* __append10(char* __first, _Tp __value) _NOEXCEPT {
  return __itoa::__append8(__itoa::__append2(__first, static_cast<uint32_t>(__value / 100000000)),
                           static_cast<uint32_t>(__value % 100000000));
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char*
__base_10_u32(char* __first, uint32_t __value) _NOEXCEPT {
  if (__value < 1000000) {
    if (__value < 10000) {
      if (__value < 100) {
        // 0 <= __value < 100
        if (__value < 10)
          return __itoa::__append1(__first, __value);
        return __itoa::__append2(__first, __value);
      }
      // 100 <= __value < 10'000
      if (__value < 1000)
        return __itoa::__append3(__first, __value);
      return __itoa::__append4(__first, __value);
    }

    // 10'000 <= __value < 1'000'000
    if (__value < 100000)
      return __itoa::__append5(__first, __value);
    return __itoa::__append6(__first, __value);
  }

  // __value => 1'000'000
  if (__value < 100000000) {
    // 1'000'000 <= __value < 100'000'000
    if (__value < 10000000)
      return __itoa::__append7(__first, __value);
    return __itoa::__append8(__first, __value);
  }

  // 100'000'000 <= __value < max
  if (__value < 1000000000)
    return __itoa::__append9(__first, __value);
  return __itoa::__append10(__first, __value);
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char*
__base_10_u64(char* __buffer, uint64_t __value) _NOEXCEPT {
  if (__value <= UINT32_MAX)
    return __itoa::__base_10_u32(__buffer, static_cast<uint32_t>(__value));

  // Numbers in the range UINT32_MAX <= val < 10'000'000'000 always contain 10
  // digits and are outputted after this if statement.
  if (__value >= 10000000000) {
    // This function properly deterimines the first non-zero leading digit.
    __buffer = __itoa::__base_10_u32(__buffer, static_cast<uint32_t>(__value / 10000000000));
    __value %= 10000000000;
  }
  return __itoa::__append10(__buffer, __value);
}

#  if _LIBCPP_HAS_INT128
/// \returns 10^\a exp
///
/// \pre \a exp [19, 39]
///
/// \note The lookup table contains a partial set of exponents limiting the
/// range that can be used. However the range is sufficient for
/// \ref __base_10_u128.
_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline __uint128_t __pow_10(int __exp) _NOEXCEPT {
  _LIBCPP_ASSERT_INTERNAL(__exp >= __pow10_128_offset, "Index out of bounds");
  return __pow10_128[__exp - __pow10_128_offset];
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char*
__base_10_u128(char* __buffer, __uint128_t __value) _NOEXCEPT {
  _LIBCPP_ASSERT_INTERNAL(
      __value > numeric_limits<uint64_t>::max(), "The optimizations for this algorithm fails when this isn't true.");

  // Unlike the 64 to 32 bit case the 128 bit case the "upper half" can't be
  // stored in the "lower half". Instead we first need to handle the top most
  // digits separately.
  //
  // Maximum unsigned values
  // 64  bit                             18'446'744'073'709'551'615 (20 digits)
  // 128 bit    340'282'366'920'938'463'463'374'607'431'768'211'455 (39 digits)
  // step 1     ^                                                   ([0-1] digits)
  // step 2      ^^^^^^^^^^^^^^^^^^^^^^^^^                          ([0-19] digits)
  // step 3                               ^^^^^^^^^^^^^^^^^^^^^^^^^ (19 digits)
  if (__value >= __itoa::__pow_10(38)) {
    // step 1
    __buffer = __itoa::__append1(__buffer, static_cast<uint32_t>(__value / __itoa::__pow_10(38)));
    __value %= __itoa::__pow_10(38);

    // step 2 always 19 digits.
    // They are handled here since leading zeros need to be appended to the buffer,
    __buffer = __itoa::__append9(__buffer, static_cast<uint32_t>(__value / __itoa::__pow_10(29)));
    __value %= __itoa::__pow_10(29);
    __buffer = __itoa::__append10(__buffer, static_cast<uint64_t>(__value / __itoa::__pow_10(19)));
    __value %= __itoa::__pow_10(19);
  } else {
    // step 2
    // This version needs to determine the position of the leading non-zero digit.
    __buffer = __base_10_u64(__buffer, static_cast<uint64_t>(__value / __itoa::__pow_10(19)));
    __value %= __itoa::__pow_10(19);
  }

  // Step 3
  __buffer = __itoa::__append9(__buffer, static_cast<uint32_t>(__value / 10000000000));
  __buffer = __itoa::__append10(__buffer, static_cast<uint64_t>(__value % 10000000000));

  return __buffer;
}
#  endif

#if _LIBCPP_HAS_INT256
/// \returns 10^\a exp
///
/// \pre \a exp [0, 77]
_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline __uint256_t __pow_10_256(int __exp) _NOEXCEPT {
  _LIBCPP_ASSERT_INTERNAL(__exp >= __pow10_256_offset, "Index out of bounds");
  return __pow10_256[__exp - __pow10_256_offset];
}

_LIBCPP_CONSTEXPR_SINCE_CXX23 _LIBCPP_HIDE_FROM_ABI inline char*
__base_10_u256(char* __buffer, __uint256_t __value) _NOEXCEPT {
  _LIBCPP_ASSERT_INTERNAL(
      __value > numeric_limits<__uint128_t>::max(), "The optimizations for this algorithm fail when this isn't true.");

  // Maximum unsigned values:
  // 128 bit                  340'282'366'920'938'463'463'374'607'431'768'211'455 (39 digits)
  // 256 bit  115'792'089'237'316'195'423'570'985'008'687'907'853'
  //          269'984'665'640'564'039'457'584'007'913'129'639'935   (78 digits)
  //
  // Strategy: divide into chunks of 19 digits (10^19 fits in uint64_t).
  // A 256-bit number has at most 78 digits = 4 chunks of 19 + 2 leading digits.
  // We peel off 19-digit chunks from the bottom using 256-bit division by 10^19.

  __uint256_t __p19 = __pow_10_256(19);

  // A 256-bit number has at most 78 digits = 5 chunks of up to 19 digits.
  // Extract 5 chunks of at most 19 digits each from the bottom.
  uint64_t __c0 = static_cast<uint64_t>(__value % __p19);
  __value /= __p19;
  uint64_t __c1 = static_cast<uint64_t>(__value % __p19);
  __value /= __p19;
  uint64_t __c2 = static_cast<uint64_t>(__value % __p19);
  __value /= __p19;
  uint64_t __c3 = static_cast<uint64_t>(__value % __p19);
  __value /= __p19;
  uint64_t __c4 = static_cast<uint64_t>(__value); // at most 2 digits

  // Emit 19-digit zero-padded chunk: [9 digits] + [10 digits]
  auto __emit_padded = [&](uint64_t __c) {
    __buffer = __itoa::__append9(__buffer, static_cast<uint32_t>(__c / 10000000000));
    __buffer = __itoa::__append10(__buffer, __c % 10000000000);
  };

  // Find the first non-zero chunk and emit it with variable width.
  if (__c4) {
    __buffer = __base_10_u64(__buffer, __c4);
    __emit_padded(__c3);
    __emit_padded(__c2);
  } else if (__c3) {
    __buffer = __base_10_u64(__buffer, __c3);
    __emit_padded(__c2);
  } else {
    __buffer = __base_10_u64(__buffer, __c2);
  }
  __emit_padded(__c1);
  __emit_padded(__c0);

  return __buffer;
}
#endif
} // namespace __itoa

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CHARCONV_TO_CHARS_BASE_10_H
