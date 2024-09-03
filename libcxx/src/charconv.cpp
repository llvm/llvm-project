//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <charconv>

#include "include/to_chars_floating_point.h"

_LIBCPP_BEGIN_NAMESPACE_STD

#ifndef _LIBCPP_ABI_DO_NOT_EXPORT_TO_CHARS_BASE_10

namespace __itoa {

_LIBCPP_EXPORTED_FROM_ABI char* __u32toa(uint32_t value, char* buffer) noexcept { return __base_10_u32(buffer, value); }

_LIBCPP_EXPORTED_FROM_ABI char* __u64toa(uint64_t value, char* buffer) noexcept { return __base_10_u64(buffer, value); }

} // namespace __itoa

#endif // _LIBCPP_ABI_DO_NOT_EXPORT_TO_CHARS_BASE_10

// The original version of floating-point to_chars was written by Microsoft and
// contributed with the following license.

// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This implementation is dedicated to the memory of Mary and Thavatchai.

to_chars_result to_chars(char* __first, char* __last, float __value) {
  return _Floating_to_chars<_Floating_to_chars_overload::_Plain>(__first, __last, __value, chars_format{}, 0);
}

to_chars_result to_chars(char* __first, char* __last, double __value) {
  return _Floating_to_chars<_Floating_to_chars_overload::_Plain>(__first, __last, __value, chars_format{}, 0);
}

to_chars_result to_chars(char* __first, char* __last, long double __value) {
  return _Floating_to_chars<_Floating_to_chars_overload::_Plain>(
      __first, __last, static_cast<double>(__value), chars_format{}, 0);
}

to_chars_result to_chars(char* __first, char* __last, float __value, chars_format __fmt) {
  return _Floating_to_chars<_Floating_to_chars_overload::_Format_only>(__first, __last, __value, __fmt, 0);
}

to_chars_result to_chars(char* __first, char* __last, double __value, chars_format __fmt) {
  return _Floating_to_chars<_Floating_to_chars_overload::_Format_only>(__first, __last, __value, __fmt, 0);
}

to_chars_result to_chars(char* __first, char* __last, long double __value, chars_format __fmt) {
  return _Floating_to_chars<_Floating_to_chars_overload::_Format_only>(
      __first, __last, static_cast<double>(__value), __fmt, 0);
}

to_chars_result to_chars(char* __first, char* __last, float __value, chars_format __fmt, int __precision) {
  return _Floating_to_chars<_Floating_to_chars_overload::_Format_precision>(
      __first, __last, __value, __fmt, __precision);
}

to_chars_result to_chars(char* __first, char* __last, double __value, chars_format __fmt, int __precision) {
  return _Floating_to_chars<_Floating_to_chars_overload::_Format_precision>(
      __first, __last, __value, __fmt, __precision);
}

to_chars_result to_chars(char* __first, char* __last, long double __value, chars_format __fmt, int __precision) {
  return _Floating_to_chars<_Floating_to_chars_overload::_Format_precision>(
      __first, __last, static_cast<double>(__value), __fmt, __precision);
}

template __to_chars_offset_result
__external_to_chars_integral<2, uint32_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, uint32_t);

template __to_chars_offset_result
__external_to_chars_integral<8, uint32_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, uint32_t);

template __to_chars_offset_result
__external_to_chars_integral<16, uint32_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, uint32_t);

template __to_chars_offset_result
__external_to_chars_integral<2, uint64_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, uint64_t);

template __to_chars_offset_result
__external_to_chars_integral<8, uint64_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, uint64_t);

template __to_chars_offset_result
__external_to_chars_integral<16, uint64_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, uint64_t);

#ifndef _LIBCPP_HAS_NO_INT128
template __to_chars_offset_result
__external_to_chars_integral<2, __uint128_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, __uint128_t);

template __to_chars_offset_result
__external_to_chars_integral<8, __uint128_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, __uint128_t);

template __to_chars_offset_result
__external_to_chars_integral<16, __uint128_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, __uint128_t);
#endif // _LIBCPP_HAS_NO_INT128

template __to_chars_offset_result
__external_to_chars_integral<uint32_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, int, uint32_t);

template __to_chars_offset_result
__external_to_chars_integral<uint64_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, int, uint64_t);

#ifndef _LIBCPP_HAS_NO_INT128
template __to_chars_offset_result
__external_to_chars_integral<__uint128_t>(_LIBCPP_NOESCAPE char*, _LIBCPP_NOESCAPE char*, int, __uint128_t);
#endif
_LIBCPP_END_NAMESPACE_STD
