//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// template <class It>
// constexpr explicit(Extent != dynamic_extent) span(It first, size_type count);
//  If Extent is not equal to dynamic_extent, then count shall be equal to Extent.
//

#include <cassert>
#include <cstddef>
#include <iterator>
#include <span>
#include <type_traits>

template <std::size_t Extent>
constexpr void test_constructibility() {
  struct Other {};
  static_assert(std::is_constructible_v<std::span<int, Extent>, int*, std::size_t>);
  static_assert(!std::is_constructible_v<std::span<int, Extent>, const int*, std::size_t>);
  static_assert(std::is_constructible_v<std::span<const int, Extent>, int*, std::size_t>);
  static_assert(std::is_constructible_v<std::span<const int, Extent>, const int*, std::size_t>);
  static_assert(!std::is_constructible_v<std::span<int, Extent>, volatile int*, std::size_t>);
  static_assert(!std::is_constructible_v<std::span<int, Extent>, const volatile int*, std::size_t>);
  static_assert(!std::is_constructible_v<std::span<const int, Extent>, volatile int*, std::size_t>);
  static_assert(!std::is_constructible_v<std::span<const int, Extent>, const volatile int*, std::size_t>);
  static_assert(!std::is_constructible_v<std::span<volatile int, Extent>, const int*, std::size_t>);
  static_assert(!std::is_constructible_v<std::span<volatile int, Extent>, const volatile int*, std::size_t>);
  static_assert(
      !std::is_constructible_v<std::span<int, Extent>, double*, std::size_t>); // iterator type differs from span type
  static_assert(!std::is_constructible_v<std::span<int, Extent>, std::size_t, size_t>);
  static_assert(!std::is_constructible_v<std::span<int, Extent>, Other*, std::size_t>); // unrelated iterator type
}

template <class T>
constexpr bool test_ctor() {
  T val[2] = {};
  auto s1 = std::span<T>(val, 2);
  auto s2 = std::span<T, 2>(val, 2);
  assert(s1.data() == std::data(val) && s1.size() == std::size(val));
  assert(s2.data() == std::data(val) && s2.size() == std::size(val));
  return true;
}

constexpr bool test() {
  test_constructibility<std::dynamic_extent>();
  test_constructibility<3>();

  struct A {};
  test_ctor<int>();
  test_ctor<A>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
