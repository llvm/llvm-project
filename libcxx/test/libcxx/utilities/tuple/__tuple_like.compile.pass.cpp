//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <tuple>

// 22.4.3 Concept `tuple-like` [tuple.like]
//
// template<class T>
//  concept tuple-like;           // exposition only

#include <array>
#include <complex>
#include <ranges>
#include <tuple>

#include "test_iterators.h"

// Non-tuple-like type

static_assert(!std::__tuple_like<int>);

// Tuple-like: array

static_assert(std::__tuple_like<std::array<int, 0>>);
static_assert(std::__tuple_like<std::array<int, 1>>);
static_assert(std::__tuple_like<std::array<int, 2>>);
static_assert(std::__tuple_like<std::array<int, 2728>>);

// Tuple-like: complex

#if _LIBCPP_STD_VER >= 26
static_assert(std::__tuple_like<std::complex<float>>);
static_assert(std::__tuple_like<std::complex<double>>);
static_assert(std::__tuple_like<std::complex<long double>>);
#endif

// Tuple-like: pair

static_assert(std::__tuple_like<std::pair<int, float>>);

// Tuple-like: tuple

static_assert(std::__tuple_like<std::tuple<int>>);
static_assert(std::__tuple_like<std::tuple<int, float>>);
static_assert(std::__tuple_like<std::tuple<int, float, double>>);

// Support for <ranges>

using FI = forward_iterator<int*>;
static_assert(std::__tuple_like<std::ranges::subrange<FI, FI, std::ranges::subrange_kind::sized>>);
static_assert(std::__tuple_like<std::ranges::subrange<FI, FI, std::ranges::subrange_kind::unsized>>);
static_assert(std::__tuple_like<std::ranges::subrange<int*, int*, std::ranges::subrange_kind::sized>>);
static_assert(std::__tuple_like<std::ranges::subrange<int*, std::nullptr_t, std::ranges::subrange_kind::unsized>>);

template <typename Iter>
void test_subrange_sized() {
  static_assert(std::__tuple_like<std::ranges::subrange<Iter, Iter, std::ranges::subrange_kind::sized>>);
}

template <typename Iter>
void test_subrange_unsized() {
  static_assert(std::__tuple_like<std::ranges::subrange<Iter, Iter, std::ranges::subrange_kind::unsized>>);
}

void test() {
  test_subrange_sized<forward_iterator<int*>>();
  test_subrange_sized<bidirectional_iterator<int*>>();
  test_subrange_sized<random_access_iterator<int*>>();
  test_subrange_sized<contiguous_iterator<int*>>();
  test_subrange_sized<int*>();

  test_subrange_sized<forward_iterator<int const*>>();
  test_subrange_sized<bidirectional_iterator<int const*>>();
  test_subrange_sized<random_access_iterator<int const*>>();
  test_subrange_sized<contiguous_iterator<int const*>>();
  test_subrange_sized<int const*>();

  test_subrange_unsized<forward_iterator<int*>>();
  test_subrange_unsized<bidirectional_iterator<int*>>();
  static_assert(std::__tuple_like<std::ranges::subrange<int*, std::nullptr_t, std::ranges::subrange_kind::unsized>>);

  test_subrange_unsized<forward_iterator<int const*>>();
  test_subrange_unsized<bidirectional_iterator<int const*>>();
  static_assert(
      std::__tuple_like<std::ranges::subrange<const int*, std::nullptr_t, std::ranges::subrange_kind::unsized>>);
}
