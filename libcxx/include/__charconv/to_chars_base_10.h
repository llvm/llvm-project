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

#include <__charconv/tables.h>
#include <__config>
#include <cstdint>
#include <cstring>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#ifndef _LIBCPP_CXX03_LANG

namespace __itoa {

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI char* __append1(char* __buffer, _Tp __value) noexcept {
  *__buffer = '0' + static_cast<char>(__value);
  return __buffer + 1;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI char* __append2(char* __buffer, _Tp __value) noexcept {
  std::memcpy(__buffer, &__digits_base_10<>::__value[(__value)*2], 2);
  return __buffer + 2;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI char* __append3(char* __buffer, _Tp __value) noexcept {
  return __itoa::__append2(__itoa::__append1(__buffer, (__value) / 100), (__value) % 100);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI char* __append4(char* __buffer, _Tp __value) noexcept {
  return __itoa::__append2(__itoa::__append2(__buffer, (__value) / 100), (__value) % 100);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI char* __append2_no_zeros(char* __buffer, _Tp __value) noexcept {
  if (__value < 10)
    return __itoa::__append1(__buffer, __value);
  else
    return __itoa::__append2(__buffer, __value);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI char* __append4_no_zeros(char* __buffer, _Tp __value) noexcept {
  if (__value < 100)
    return __itoa::__append2_no_zeros(__buffer, __value);
  else if (__value < 1000)
    return __itoa::__append3(__buffer, __value);
  else
    return __itoa::__append4(__buffer, __value);
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI char* __append8_no_zeros(char* __buffer, _Tp __value) noexcept {
  if (__value < 10000)
    __buffer = __itoa::__append4_no_zeros(__buffer, __value);
  else {
    __buffer = __itoa::__append4_no_zeros(__buffer, __value / 10000);
    __buffer = __itoa::__append4(__buffer, __value % 10000);
  }
  return __buffer;
}

_LIBCPP_HIDE_FROM_ABI inline char* __base_10_u32(uint32_t __value, char* __buffer) noexcept {
  if (__value < 100000000)
    __buffer = __itoa::__append8_no_zeros(__buffer, __value);
  else {
    // __value = aabbbbcccc in decimal
    const uint32_t __a = __value / 100000000; // 1 to 42
    __value %= 100000000;

    __buffer = __itoa::__append2_no_zeros(__buffer, __a);
    __buffer = __itoa::__append4(__buffer, __value / 10000);
    __buffer = __itoa::__append4(__buffer, __value % 10000);
  }

  return __buffer;
}

_LIBCPP_HIDE_FROM_ABI inline char* __base_10_u64(uint64_t __value, char* __buffer) noexcept {
  if (__value < 100000000)
    __buffer = __itoa::__append8_no_zeros(__buffer, static_cast<uint32_t>(__value));
  else if (__value < 10000000000000000) {
    const uint32_t __v0 = static_cast<uint32_t>(__value / 100000000);
    const uint32_t __v1 = static_cast<uint32_t>(__value % 100000000);

    __buffer = __itoa::__append8_no_zeros(__buffer, __v0);
    __buffer = __itoa::__append4(__buffer, __v1 / 10000);
    __buffer = __itoa::__append4(__buffer, __v1 % 10000);
  } else {
    const uint32_t __a = static_cast<uint32_t>(__value / 10000000000000000); // 1 to 1844
    __value %= 10000000000000000;

    __buffer = __itoa::__append4_no_zeros(__buffer, __a);

    const uint32_t __v0 = static_cast<uint32_t>(__value / 100000000);
    const uint32_t __v1 = static_cast<uint32_t>(__value % 100000000);
    __buffer = __itoa::__append4(__buffer, __v0 / 10000);
    __buffer = __itoa::__append4(__buffer, __v0 % 10000);
    __buffer = __itoa::__append4(__buffer, __v1 / 10000);
    __buffer = __itoa::__append4(__buffer, __v1 % 10000);
  }

  return __buffer;
}

} // namespace __itoa

#endif // _LIBCPP_CXX03_LANG

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CHARCONV_TO_CHARS_BASE_10_H
