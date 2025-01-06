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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <span> // dynamic_extent
#include <type_traits>

#include "test_macros.h"

#include "../ConvertibleToIntegral.h"

template<class Mapping, class ... Indices>
concept operator_constraints = requires(Mapping m, Indices ... idxs) {
  {std::is_same_v<decltype(m(idxs...)), typename Mapping::index_type>};
};

template<class Mapping, class ... Indices>
  requires(
    operator_constraints<Mapping, Indices...>
  )
constexpr bool check_operator_constraints(Mapping m, Indices ... idxs) {
  (void) m(idxs...);
  return true;
}

template<class Mapping, class ... Indices>
constexpr bool check_operator_constraints(Mapping, Indices ...) {
  return false;
}

template <class M, class T, class... Args>
constexpr void iterate_right(M m, T& count, Args... args) {
  constexpr size_t r = sizeof...(Args);
  if constexpr (M::extents_type::rank() == r) {
    ASSERT_NOEXCEPT(m(args...));
    assert(count == m(args...));
    count++;
  } else {
    for (typename M::index_type i = 0; i < m.extents().extent(r); i++) {
      iterate_right(m, count, args..., i);
    }
  }
}

template <class E, class... Args>
constexpr void test_iteration(Args... args) {
  using M = std::layout_right::mapping<E>;
  M m(E(args...));

  typename E::index_type count = 0;
  iterate_right(m, count);
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;
  test_iteration<std::extents<int>>();
  test_iteration<std::extents<unsigned, D>>(1);
  test_iteration<std::extents<unsigned, D>>(7);
  test_iteration<std::extents<unsigned, 7>>();
  test_iteration<std::extents<unsigned, 7, 8>>();
  test_iteration<std::extents<signed char, D, D, D, D>>(1, 1, 1, 1);

  // Check operator constraint for number of arguments
  static_assert(check_operator_constraints(std::layout_right::mapping<std::extents<int, D>>(std::extents<int, D>(1)), 0));
  static_assert(!check_operator_constraints(std::layout_right::mapping<std::extents<int, D>>(std::extents<int, D>(1)), 0, 0));

  // Check operator constraint for convertibility of arguments to index_type
  static_assert(check_operator_constraints(std::layout_right::mapping<std::extents<int, D>>(std::extents<int, D>(1)), IntType(0)));
  static_assert(!check_operator_constraints(std::layout_right::mapping<std::extents<unsigned, D>>(std::extents<unsigned, D>(1)), IntType(0)));

  // Check operator constraint for no-throw-constructibility of index_type from arguments
  static_assert(!check_operator_constraints(std::layout_right::mapping<std::extents<unsigned char, D>>(std::extents<unsigned char, D>(1)), IntType(0)));

  return true;
}

constexpr bool test_large() {
  constexpr size_t D = std::dynamic_extent;
  test_iteration<std::extents<int64_t, D, 8, D, D>>(7, 9, 10);
  test_iteration<std::extents<int64_t, D, 8, 1, D>>(7, 10);
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
