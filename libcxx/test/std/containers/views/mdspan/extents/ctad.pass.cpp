//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// template <class... Integrals>
// explicit extents(Integrals...) -> see below;
//   Constraints: (is_convertible_v<Integrals, size_t> && ...) is true.
//
// Remarks: The deduced type is dextents<size_t, sizeof...(Integrals)>.           // until C++26
// Remarks: The deduced type is extents<size_t, maybe-static-ext<Integrals>...>.  // since C++26

#include <mdspan>
#include <cassert>
#include <span> // dynamic_extent
#include <type_traits>

#include "../ConvertibleToIntegral.h"
#include "test_macros.h"

struct NoDefaultCtorIndex {
  size_t value;
  constexpr NoDefaultCtorIndex() = delete;
  constexpr NoDefaultCtorIndex(size_t val) : value(val) {}
  constexpr operator size_t() const noexcept { return value; }
};

template <class E, class Expected>
constexpr void test(E e, Expected expected) {
  ASSERT_SAME_TYPE(E, Expected);
  assert(e == expected);
}

constexpr bool test() {
  constexpr std::size_t D = std::dynamic_extent;

  test(std::extents(), std::extents<size_t>());
  test(std::extents(1), std::extents<std::size_t, D>(1));
  test(std::extents(1, 2u), std::extents<std::size_t, D, D>(1, 2u));
  test(std::extents(1, 2u, 3, 4, 5, 6, 7, 8, 9),
       std::extents<std::size_t, D, D, D, D, D, D, D, D, D>(1, 2u, 3, 4, 5, 6, 7, 8, 9));
  test(std::extents(NoDefaultCtorIndex{1}, NoDefaultCtorIndex{2}), std::extents<std::size_t, D, D>(1, 2));

#if _LIBCPP_STD_VER >= 26
  // P3029R1: deduction from `integral_constant`
  test(std::extents(std::integral_constant<size_t, 5>{}), std::extents<std::size_t, 5>());
  test(std::extents(std::integral_constant<size_t, 5>{}, 6), std::extents<std::size_t, 5, std::dynamic_extent>(6));
  test(std::extents(std::integral_constant<size_t, 5>{}, 6, std::integral_constant<size_t, 7>{}),
       std::extents<std::size_t, 5, std::dynamic_extent, 7>(6));
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
