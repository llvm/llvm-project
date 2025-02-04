//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <mdspan>

// Test default iteration:
//
// template<class... Indices>
//   constexpr index_type operator()(Indices...) const noexcept;
//
// Constraints:
//   * sizeof...(Indices) == extents_type::rank() is true,
//   * (is_convertible_v<Indices, index_type> && ...) is true, and
//   * (is_nothrow_constructible_v<index_type, Indices> && ...) is true.
//
// Preconditions:
//   * extents_type::index-cast(i) is a multidimensional index in extents_.

#include <mdspan>
#include <array>
#include <cassert>
#include <cstdint>
#include <span> // dynamic_extent
#include <type_traits>

#include "test_macros.h"

#include "../ConvertibleToIntegral.h"

template <class Mapping, class... Indices>
concept operator_constraints = requires(Mapping m, Indices... idxs) {
  { std::is_same_v<decltype(m(idxs...)), typename Mapping::index_type> };
};

template <class Mapping, class... Indices>
  requires(operator_constraints<Mapping, Indices...>)
constexpr bool check_operator_constraints(Mapping m, Indices... idxs) {
  (void)m(idxs...);
  return true;
}

template <class Mapping, class... Indices>
constexpr bool check_operator_constraints(Mapping, Indices...) {
  return false;
}

template <class M, class... Args>
constexpr void iterate_stride(M m, const std::array<int, M::extents_type::rank()>& strides, Args... args) {
  constexpr int r = static_cast<int>(M::extents_type::rank()) - 1 - static_cast<int>(sizeof...(Args));
  if constexpr (-1 == r) {
    ASSERT_NOEXCEPT(m(args...));
    std::size_t expected_val = static_cast<std::size_t>([&]<std::size_t... Pos>(std::index_sequence<Pos...>) {
      return ((args * strides[Pos]) + ... + 0);
    }(std::make_index_sequence<M::extents_type::rank()>()));
    assert(expected_val == static_cast<std::size_t>(m(args...)));
  } else {
    for (typename M::index_type i = 0; i < m.extents().extent(r); i++) {
      iterate_stride(m, strides, i, args...);
    }
  }
}

template <class E, class... Args>
constexpr void test_iteration(std::array<int, E::rank()> strides, Args... args) {
  using M = std::layout_stride::mapping<E>;
  M m(E(args...), strides);

  iterate_stride(m, strides);
}

constexpr bool test() {
  constexpr std::size_t D = std::dynamic_extent;
  test_iteration<std::extents<int>>(std::array<int, 0>{});
  test_iteration<std::extents<unsigned, D>>(std::array<int, 1>{2}, 1);
  test_iteration<std::extents<unsigned, D>>(std::array<int, 1>{3}, 7);
  test_iteration<std::extents<unsigned, 7>>(std::array<int, 1>{4});
  test_iteration<std::extents<unsigned, 7, 8>>(std::array<int, 2>{25, 3});
  test_iteration<std::extents<signed char, D, D, D, D>>(std::array<int, 4>{1, 1, 1, 1}, 1, 1, 1, 1);

  // Check operator constraint for number of arguments
  static_assert(check_operator_constraints(
      std::layout_stride::mapping<std::extents<int, D>>(std::extents<int, D>(1), std::array{1}), 0));
  static_assert(!check_operator_constraints(
      std::layout_stride::mapping<std::extents<int, D>>(std::extents<int, D>(1), std::array{1}), 0, 0));

  // Check operator constraint for convertibility of arguments to index_type
  static_assert(check_operator_constraints(
      std::layout_stride::mapping<std::extents<int, D>>(std::extents<int, D>(1), std::array{1}), IntType(0)));
  static_assert(!check_operator_constraints(
      std::layout_stride::mapping<std::extents<unsigned, D>>(std::extents<unsigned, D>(1), std::array{1}), IntType(0)));

  // Check operator constraint for no-throw-constructibility of index_type from arguments
  static_assert(!check_operator_constraints(
      std::layout_stride::mapping<std::extents<unsigned char, D>>(std::extents<unsigned char, D>(1), std::array{1}),
      IntType(0)));

  return true;
}

constexpr bool test_large() {
  constexpr std::size_t D = std::dynamic_extent;
  test_iteration<std::extents<int64_t, D, 8, D, D>>(std::array<int, 4>{2000, 2, 20, 200}, 7, 9, 10);
  test_iteration<std::extents<int64_t, D, 8, 1, D>>(std::array<int, 4>{2000, 20, 20, 200}, 7, 10);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  // The large test iterates over ~10k loop indices.
  // With assertions enabled this triggered the maximum default limit
  // for steps in consteval expressions. Assertions roughly double the
  // total number of instructions, so this was already close to the maximum.
  test_large();
  return 0;
}
