//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <bit>

// Regression test for [bit] templates constrained on __unsigned_integer:
// cv-qualified types are not unsigned integer types per [basic.fundamental]
// /p2 and must be rejected by the SFINAE constraint. Explicit template args
// bypass the by-value deduction strip, so the rejection has to live in the
// constraint, not just at call site.
//
// std::byteswap is NOT in this list: its Constraints clause is `T models
// integral`, which admits cv-qualified types.

#include <bit>

template <class T>
concept _has_bit_ceil = requires { std::bit_ceil<T>(T{}); };
template <class T>
concept _has_bit_floor = requires { std::bit_floor<T>(T{}); };
template <class T>
concept _has_bit_width = requires { std::bit_width<T>(T{}); };
template <class T>
concept _has_has_single_bit = requires { std::has_single_bit<T>(T{}); };
template <class T>
concept _has_rotl = requires { std::rotl<T>(T{}, 0); };
template <class T>
concept _has_rotr = requires { std::rotr<T>(T{}, 0); };
template <class T>
concept _has_countl_zero = requires { std::countl_zero<T>(T{}); };
template <class T>
concept _has_countl_one = requires { std::countl_one<T>(T{}); };
template <class T>
concept _has_countr_zero = requires { std::countr_zero<T>(T{}); };
template <class T>
concept _has_countr_one = requires { std::countr_one<T>(T{}); };
template <class T>
concept _has_popcount = requires { std::popcount<T>(T{}); };

template <class T>
constexpr void check_admitted() {
  static_assert(_has_bit_ceil<T>);
  static_assert(_has_bit_floor<T>);
  static_assert(_has_bit_width<T>);
  static_assert(_has_has_single_bit<T>);
  static_assert(_has_rotl<T>);
  static_assert(_has_rotr<T>);
  static_assert(_has_countl_zero<T>);
  static_assert(_has_countl_one<T>);
  static_assert(_has_countr_zero<T>);
  static_assert(_has_countr_one<T>);
  static_assert(_has_popcount<T>);
}

template <class T>
constexpr void check_rejected() {
  static_assert(!_has_bit_ceil<T>);
  static_assert(!_has_bit_floor<T>);
  static_assert(!_has_bit_width<T>);
  static_assert(!_has_has_single_bit<T>);
  static_assert(!_has_rotl<T>);
  static_assert(!_has_rotr<T>);
  static_assert(!_has_countl_zero<T>);
  static_assert(!_has_countl_one<T>);
  static_assert(!_has_countr_zero<T>);
  static_assert(!_has_countr_one<T>);
  static_assert(!_has_popcount<T>);
}

// Unqualified unsigned integer types pass.
template void check_admitted<unsigned int>();
template void check_admitted<unsigned long>();
template void check_admitted<unsigned long long>();

// cv-qualified versions of unsigned integer types are rejected.
template void check_rejected<const unsigned int>();
template void check_rejected<volatile unsigned int>();
template void check_rejected<const volatile unsigned int>();
template void check_rejected<const unsigned long>();
template void check_rejected<const unsigned long long>();

// Signed and character types stay rejected.
template void check_rejected<int>();
template void check_rejected<const int>();
template void check_rejected<bool>();
template void check_rejected<const bool>();
template void check_rejected<char>();
template void check_rejected<const char>();
