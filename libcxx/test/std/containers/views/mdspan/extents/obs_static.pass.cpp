//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// static constexpr rank_type rank() noexcept;
// static constexpr rank_type rank_dynamic() noexcept;
//
// static constexpr size_t static_extent(rank_type i) noexcept;
//
//   Preconditions: i < rank() is true.
//
//   Returns: Ei.
//
//
// constexpr index_type extent(rank_type i) const noexcept;
//
//   Preconditions: i < rank() is true.
//
//   Returns: Di.
//

#include <mdspan>
#include <cassert>
#include <utility>

#include "test_macros.h"

template <class E, size_t rank, size_t rank_dynamic, size_t... StaticExts, size_t... Indices>
void test_static_observers(std::index_sequence<StaticExts...>, std::index_sequence<Indices...>) {
  ASSERT_NOEXCEPT(E::rank());
  static_assert(E::rank() == rank);
  ASSERT_NOEXCEPT(E::rank_dynamic());
  static_assert(E::rank_dynamic() == rank_dynamic);

  // Let's only test this if the call isn't a precondition violation
  if constexpr (rank > 0) {
    ASSERT_NOEXCEPT(E::static_extent(0));
    ASSERT_SAME_TYPE(decltype(E::static_extent(0)), size_t);
    static_assert(((E::static_extent(Indices) == StaticExts) && ...));
  }
}

template <class E, size_t rank, size_t rank_dynamic, size_t... StaticExts>
void test_static_observers() {
  test_static_observers<E, rank, rank_dynamic>(
      std::index_sequence<StaticExts...>(), std::make_index_sequence<sizeof...(StaticExts)>());
}

template <class T>
void test() {
  constexpr size_t D = std::dynamic_extent;
  constexpr size_t S = 5;

  test_static_observers<std::extents<T>, 0, 0>();

  test_static_observers<std::extents<T, S>, 1, 0, S>();
  test_static_observers<std::extents<T, D>, 1, 1, D>();

  test_static_observers<std::extents<T, S, S>, 2, 0, S, S>();
  test_static_observers<std::extents<T, S, D>, 2, 1, S, D>();
  test_static_observers<std::extents<T, D, S>, 2, 1, D, S>();
  test_static_observers<std::extents<T, D, D>, 2, 2, D, D>();

  test_static_observers<std::extents<T, S, S, S>, 3, 0, S, S, S>();
  test_static_observers<std::extents<T, S, S, D>, 3, 1, S, S, D>();
  test_static_observers<std::extents<T, S, D, S>, 3, 1, S, D, S>();
  test_static_observers<std::extents<T, D, S, S>, 3, 1, D, S, S>();
  test_static_observers<std::extents<T, S, D, D>, 3, 2, S, D, D>();
  test_static_observers<std::extents<T, D, S, D>, 3, 2, D, S, D>();
  test_static_observers<std::extents<T, D, D, S>, 3, 2, D, D, S>();
  test_static_observers<std::extents<T, D, D, D>, 3, 3, D, D, D>();
}

int main(int, char**) {
  test<int>();
  test<unsigned>();
  test<signed char>();
  test<long long>();
  test<size_t>();
  return 0;
}
