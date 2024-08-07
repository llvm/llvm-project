//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// class iterator

#include <cassert>
#include <concepts>
#include <iterator>
#include <span>
#include <string>
#include <version> // __cpp_lib_ranges_as_const is not defined in span.

#include "test_macros.h"

template <class T>
constexpr void test_type() {
  using C = std::span<T>;
  typename C::iterator ii1{}, ii2{};
  typename C::iterator ii4 = ii1;
#if TEST_STD_VER >= 23
  typename C::const_iterator cii{};
#endif
  assert(ii1 == ii2);
  assert(ii1 == ii4);
#if TEST_STD_VER >= 23
  assert(ii1 == cii);
#endif

  assert(!(ii1 != ii2));
#if TEST_STD_VER >= 23
  assert(!(ii1 != cii));
#endif

  T v;
  C c{&v, 1};
  assert(c.begin() == std::begin(c));
  assert(c.rbegin() == std::rbegin(c));
#if TEST_STD_VER >= 23
  assert(c.cbegin() == std::cbegin(c));
  assert(c.crbegin() == std::crbegin(c));
#endif

  assert(c.end() == std::end(c));
  assert(c.rend() == std::rend(c));
#if TEST_STD_VER >= 23
  assert(c.cend() == std::cend(c));
  assert(c.crend() == std::crend(c));
#endif

  assert(std::begin(c) != std::end(c));
  assert(std::rbegin(c) != std::rend(c));
#if TEST_STD_VER >= 23
  assert(std::cbegin(c) != std::cend(c));
  assert(std::crbegin(c) != std::crend(c));
#endif

  // P1614 + LWG3352
  std::same_as<std::strong_ordering> decltype(auto) r1 = ii1 <=> ii2;
  assert(r1 == std::strong_ordering::equal);

#if TEST_STD_VER >= 23
  std::same_as<std::strong_ordering> decltype(auto) r2 = cii <=> ii2;
  assert(r2 == std::strong_ordering::equal);
#endif
}

constexpr bool test() {
  test_type<char>();
  test_type<int>();
  test_type<std::string>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test(), "");

  return 0;
}
