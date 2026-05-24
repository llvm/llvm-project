//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// template<class... Indices>
// constexpr index_type operator()(Indices...) const noexcept;

#include <cassert>
#include <cstddef>
#include <mdspan>
#include <type_traits>

#include "test_macros.h"

#include "../ConvertibleToIntegral.h"

template <class Mapping, class... Indices>
concept operator_constraints = requires(Mapping mapping, Indices... idxs) {
  { std::is_same_v<decltype(mapping(idxs...)), typename Mapping::index_type> };
};

template <class Mapping, class... Indices>
  requires(operator_constraints<Mapping, Indices...>)
constexpr bool check_operator_constraints(Mapping mapping, Indices... idxs) {
  (void)mapping(idxs...);
  return true;
}

template <class Mapping, class... Indices>
constexpr bool check_operator_constraints(Mapping, Indices...) {
  return false;
}

template <class Mapping, class... Args>
constexpr void iterate_left_padded(Mapping mapping, typename Mapping::index_type expected, Args... args) {
  constexpr int r = static_cast<int>(Mapping::extents_type::rank()) - 1 - static_cast<int>(sizeof...(Args));
  if constexpr (-1 == r) {
    ASSERT_NOEXCEPT(mapping(args...));
    assert(expected == mapping(args...));
  } else {
    for (typename Mapping::index_type i = 0; i < mapping.extents().extent(r); ++i)
      iterate_left_padded(
          mapping, static_cast<typename Mapping::index_type>(expected + i * mapping.stride(r)), i, args...);
  }
}

template <class Extents, size_t PaddingValue, class... Args>
constexpr void test_iteration(Args... args) {
  using Mapping = typename std::layout_left_padded<PaddingValue>::template mapping<Extents>;
  Mapping mapping(Extents(args...));
  iterate_left_padded(mapping, typename Extents::index_type(0));
}

template <class Mapping>
constexpr void test_iteration(Mapping mapping) {
  iterate_left_padded(mapping, typename Mapping::index_type(0));
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  test_iteration<std::extents<int>, 4>();
  test_iteration<std::extents<unsigned, 7>, 4>();
  test_iteration<std::extents<unsigned, 5, 7>, 4>();
  test_iteration<std::extents<signed char, D, 2, 3>, 4>(3);

  test_iteration<std::extents<int>, D>();
  test_iteration<std::extents<unsigned, D>, D>(7);
  test_iteration(std::layout_left_padded<D>::mapping<std::extents<unsigned, 5, 7>>(std::extents<unsigned, 5, 7>(), 6));
  test_iteration(
      std::layout_left_padded<D>::mapping<std::extents<unsigned, D, 2, 3>>(std::extents<unsigned, D, 2, 3>(3), 4));

  // Check operator constraint for number of arguments
  static_assert(check_operator_constraints(
      std::layout_left_padded<D>::mapping<std::extents<int, D>>(std::extents<int, D>(1), 1), 0));
  static_assert(!check_operator_constraints(
      std::layout_left_padded<D>::mapping<std::extents<int, D>>(std::extents<int, D>(1), 1), 0, 0));

  // Check operator constraint for convertibility of arguments to index_type
  static_assert(check_operator_constraints(
      std::layout_left_padded<D>::mapping<std::extents<int, D>>(std::extents<int, D>(1), 1), IntType(0)));
  static_assert(!check_operator_constraints(
      std::layout_left_padded<D>::mapping<std::extents<unsigned, D>>(std::extents<unsigned, D>(1), 1), IntType(0)));

  // Check operator constraint for no-throw-constructibility of index_type from arguments
  static_assert(!check_operator_constraints(
      std::layout_left_padded<D>::mapping<std::extents<unsigned char, D>>(std::extents<unsigned char, D>(1), 1),
      IntType(0)));

  static_assert(check_operator_constraints(
      std::layout_left_padded<4>::mapping<std::extents<int, 2, 2>>(std::extents<int, 2, 2>()),
      RValueInt{0},
      RValueInt{1}));

  {
    constexpr std::layout_left_padded<4>::mapping<std::extents<int, 2, 2>> mapping(std::extents<int, 2, 2>{});
    static_assert(mapping(RValueInt{0}, RValueInt{1}) == 4);
  }

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
