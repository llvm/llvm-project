//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// "support/constexpr_hash.h"

#include "constexpr_hash.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "test_macros.h"

#include "poisoned_hash_helper.h"

template <class T>
constexpr void test() {
  test_hash_disabled<const T, support::constexpr_hash<const T>>();
  test_hash_disabled<volatile T, support::constexpr_hash<volatile T>>();
  test_hash_disabled<const volatile T, support::constexpr_hash<const volatile T>>();

  typedef support::constexpr_hash<T> H;

  ASSERT_NOEXCEPT(H()(T()));
  H h;

  for (int i = 0; i <= 5; ++i) {
    T t(static_cast<T>(i));
    const bool small = std::integral_constant<bool, sizeof(T) <= sizeof(std::size_t)>::value; // avoid compiler warnings
    if (small) {
      const std::size_t result = h(t);
      LIBCPP_ASSERT(result == static_cast<std::size_t>(t));
      ((void)result); // Prevent unused warning
    } else {
      T t2(static_cast<T>(i + 1));
      const std::size_t result1 = h(t);
      const std::size_t result2 = h(t2);
      LIBCPP_ASSERT(result1 != result2);
    }
  }
}

constexpr bool test_with_hash() {
  test<bool>();
  test<char>();
  test<signed char>();
  test<unsigned char>();
  test<char16_t>();
  test<char32_t>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif
  test<short>();
  test<unsigned short>();
  test<int>();
  test<unsigned int>();
  test<long>();
  test<unsigned long>();
  test<long long>();
  test<unsigned long long>();

  //  LWG #2119
  test<std::ptrdiff_t>();
  test<std::size_t>();

  test<std::int8_t>();
  test<std::int16_t>();
  test<std::int32_t>();
  test<std::int64_t>();

  test<std::int_fast8_t>();
  test<std::int_fast16_t>();
  test<std::int_fast32_t>();
  test<std::int_fast64_t>();

  test<std::int_least8_t>();
  test<std::int_least16_t>();
  test<std::int_least32_t>();
  test<std::int_least64_t>();

  test<std::intmax_t>();
  test<std::intptr_t>();

  test<std::uint8_t>();
  test<std::uint16_t>();
  test<std::uint32_t>();
  test<std::uint64_t>();

  test<std::uint_fast8_t>();
  test<std::uint_fast16_t>();
  test<std::uint_fast32_t>();
  test<std::uint_fast64_t>();

  test<std::uint_least8_t>();
  test<std::uint_least16_t>();
  test<std::uint_least32_t>();
  test<std::uint_least64_t>();

  test<std::uintmax_t>();
  test<std::uintptr_t>();

#ifndef TEST_HAS_NO_INT128
  test<__int128_t>();
  test<__uint128_t>();
#endif

  return true;
}

int main(int, char**) {
  assert(test_with_hash());

  static_assert(test_with_hash());

  return 0;
}
