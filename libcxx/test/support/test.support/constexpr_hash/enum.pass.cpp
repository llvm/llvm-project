//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// "support/constexpr_hash.h"

// Make sure that we can hash enumeration values.

#include "constexpr_hash.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "test_macros.h"

#include "poisoned_hash_helper.h"

enum class Colors { red, orange, yellow, green, blue, indigo, violet };
enum class Cardinals { zero, one, two, three, five = 5 };
enum class LongColors : short { red, orange, yellow, green, blue, indigo, violet };
enum class ShortColors : long { red, orange, yellow, green, blue, indigo, violet };
enum class EightBitColors : std::uint8_t { red, orange, yellow, green, blue, indigo, violet };

enum Fruits { apple, pear, grape, mango, cantaloupe };

template <class T>
constexpr void test() {
  test_hash_disabled<const T, support::constexpr_hash<const T>>();
  test_hash_disabled<volatile T, support::constexpr_hash<volatile T>>();
  test_hash_disabled<const volatile T, support::constexpr_hash<const volatile T>>();

  typedef support::constexpr_hash<T> H;

  ASSERT_NOEXCEPT(H()(T()));
  typedef typename std::underlying_type<T>::type under_type;

  H h1;
  support::constexpr_hash<under_type> h2;
  for (int i = 0; i <= 5; ++i) {
    T t(static_cast<T>(i));
    const bool small = std::integral_constant<bool, sizeof(T) <= sizeof(std::size_t)>::value; // avoid compiler warnings
    if (small)
      assert(h1(t) == h2(static_cast<under_type>(i)));
  }
}

constexpr bool test_with_hash() {
  test<Cardinals>();

  test<Colors>();
  test<ShortColors>();
  test<LongColors>();
  test<EightBitColors>();

  test<Fruits>();

  return true;
}

int main(int, char**) {
  assert(test_with_hash());

  static_assert(test_with_hash());

  return 0;
}
