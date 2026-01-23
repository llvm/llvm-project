//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <mdspan>
//
// template<class OtherElementType, size_t OtherByteAlignment>
//   constexpr aligned_accessor(aligned_accessor<OtherElementType, OtherByteAlignment>) noexcept {}
//
// Constraints:
//   - is_convertible_v<OtherElementType(*)[], element_type(*)[]> is true.
//   - OtherByteAlignment >= byte_alignment is true.

#include <mdspan>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"

#include "../MinimalElementType.h"

struct Base {};
struct Derived : public Base {};

template <class FromT, std::size_t FromN, class ToT, std::size_t ToN>
constexpr void test_conversion() {
  std::aligned_accessor<FromT, FromN> acc_from;
  ASSERT_NOEXCEPT(std::aligned_accessor<ToT, ToN>(acc_from));
  [[maybe_unused]] std::aligned_accessor<ToT, ToN> acc_to(acc_from);
}

template <class From, class To>
constexpr void test_it() {
  constexpr std::size_t N = alignof(From);
  static_assert(N == alignof(To));

  test_conversion<From, N, To, N>();
  test_conversion<From, 2 * N, To, N>();
  test_conversion<From, 4 * N, To, N>();
  test_conversion<From, 8 * N, To, N>();
  test_conversion<From, 16 * N, To, N>();
  test_conversion<From, 32 * N, To, N>();

  test_conversion<From, 2 * N, To, 2 * N>();
  test_conversion<From, 4 * N, To, 2 * N>();
  test_conversion<From, 8 * N, To, 2 * N>();
  test_conversion<From, 16 * N, To, 2 * N>();
  test_conversion<From, 32 * N, To, 2 * N>();

  test_conversion<From, 4 * N, To, 4 * N>();
  test_conversion<From, 8 * N, To, 4 * N>();
  test_conversion<From, 16 * N, To, 4 * N>();
  test_conversion<From, 32 * N, To, 4 * N>();

  test_conversion<From, 8 * N, To, 8 * N>();
  test_conversion<From, 16 * N, To, 8 * N>();
  test_conversion<From, 32 * N, To, 8 * N>();

  test_conversion<From, 16 * N, To, 16 * N>();
  test_conversion<From, 32 * N, To, 16 * N>();

  test_conversion<From, 32 * N, To, 32 * N>();
}

constexpr bool test() {
  test_it<int, int>();
  test_it<int, const int>();
  test_it<const int, const int>();
  test_it<MinimalElementType, MinimalElementType>();
  test_it<MinimalElementType, const MinimalElementType>();
  test_it<const MinimalElementType, const MinimalElementType>();

  // char is convertible to int, but accessors are not
  static_assert(!std::is_constructible_v<std::aligned_accessor<int, 4>, std::aligned_accessor<char, 4>>);
  // don't allow conversion from const elements to non-const
  static_assert(!std::is_constructible_v<std::aligned_accessor<int, 4>, std::aligned_accessor<const int, 4>>);
  // don't allow conversion from less to more alignment
  static_assert(!std::is_constructible_v<std::aligned_accessor<int, 8>, std::aligned_accessor<int, 4>>);
  static_assert(!std::is_constructible_v<std::aligned_accessor<const int, 8>, std::aligned_accessor<const int, 4>>);
  // MinimalElementType is constructible from int, but accessors should not be convertible
  static_assert(!std::is_constructible_v<std::aligned_accessor<MinimalElementType, 8>, std::aligned_accessor<int, 8>>);
  // don't allow conversion from const elements to non-const
  static_assert(!std::is_constructible_v<std::aligned_accessor<MinimalElementType, 16>,
                                         std::aligned_accessor<const MinimalElementType, 16>>);
  // don't allow conversion from Base to Derived
  static_assert(!std::is_constructible_v<std::aligned_accessor<Derived, 1>, std::aligned_accessor<Base, 1>>);
  // don't allow conversion from Derived to Base
  static_assert(!std::is_constructible_v<std::aligned_accessor<Base, 1>, std::aligned_accessor<Derived, 1>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
