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
#include <__charconv/tables.h>
#include <__config>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#ifndef _LIBCPP_CXX03_LANG

namespace __itoa {

_LIBCPP_HIDE_FROM_ABI inline char* __append1(char* __first, uint32_t __value) noexcept {
  *__first = '0' + static_cast<char>(__value);
  return __first + 1;
}

_LIBCPP_HIDE_FROM_ABI inline char* __append2(char* __first, uint32_t __value) noexcept {
  return std::copy_n(&__table<>::__digits_base_10[__value * 2], 2, __first);
}

_LIBCPP_HIDE_FROM_ABI inline char* __append3(char* __first, uint32_t __value) noexcept {
  return __itoa::__append2(__itoa::__append1(__first, __value / 100), __value % 100);
}

_LIBCPP_HIDE_FROM_ABI inline char* __append4(char* __first, uint32_t __value) noexcept {
  return __itoa::__append2(__itoa::__append2(__first, __value / 100), __value % 100);
}

_LIBCPP_HIDE_FROM_ABI inline char* __append5(char* __first, uint32_t __value) noexcept {
  return __itoa::__append4(__itoa::__append1(__first, __value / 10000), __value % 10000);
}

_LIBCPP_HIDE_FROM_ABI inline char* __append6(char* __first, uint32_t __value) noexcept {
  return __itoa::__append4(__itoa::__append2(__first, __value / 10000), __value % 10000);
}

_LIBCPP_HIDE_FROM_ABI inline char* __append7(char* __first, uint32_t __value) noexcept {
  return __itoa::__append6(__itoa::__append1(__first, __value / 1000000), __value % 1000000);
}

_LIBCPP_HIDE_FROM_ABI inline char* __append8(char* __first, uint32_t __value) noexcept {
  return __itoa::__append6(__itoa::__append2(__first, __value / 1000000), __value % 1000000);
}

_LIBCPP_HIDE_FROM_ABI inline char* __append9(char* __first, uint32_t __value) noexcept {
  return __itoa::__append8(__itoa::__append1(__first, __value / 100000000), __value % 100000000);
}

// This function is used for uint32_t and uint64_t.
template <class _Tp>
_LIBCPP_HIDE_FROM_ABI char* __append10(char* __first, _Tp __value) noexcept {
  return __itoa::__append8(__itoa::__append2(__first, static_cast<uint32_t>(__value / 100000000)),
                           static_cast<uint32_t>(__value % 100000000));
}

_LIBCPP_HIDE_FROM_ABI inline char* __base_10_u32(char* __first, uint32_t __value) noexcept {
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

_LIBCPP_HIDE_FROM_ABI inline char* __base_10_u64(char* __buffer, uint64_t __value) noexcept {
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

} // namespace __itoa

#endif // _LIBCPP_CXX03_LANG

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CHARCONV_TO_CHARS_BASE_10_H
