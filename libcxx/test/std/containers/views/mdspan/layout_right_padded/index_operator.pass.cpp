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
#include <concepts>
#include <cstddef>
#include <mdspan>
#include <utility>

#include "test_macros.h"

#include "../ConvertibleToIntegral.h"

template <class Mapping, class... Indices>
concept operator_constraints = requires(const Mapping& mapping, Indices... idxs) {
  { mapping(idxs...) } noexcept -> std::same_as<typename Mapping::index_type>;
};

template <class Mapping, class... Indices>
constexpr Mapping::index_type expected_result(const Mapping& mapping, Indices... idxs) {
  using index_type = Mapping::index_type;

  static_assert(sizeof...(Indices) == Mapping::extents_type::rank());

  index_type result = 0;
  size_t r          = 0;

  ((result = static_cast<index_type>(result + static_cast<index_type>(std::move(idxs)) * mapping.stride(r++))), ...);

  return result;
}

template <class Mapping, class... Indices>
constexpr void check_operator_result(Mapping mapping, Indices... idxs) {
  if constexpr (constexpr size_t rank = Mapping::extents_type::rank(); sizeof...(Indices) == rank) {
    assert(mapping(idxs...) == expected_result(mapping, idxs...));
  } else {
    constexpr size_t r = rank - 1 - sizeof...(Indices);
    for (typename Mapping::index_type i = 0; i < mapping.extents().extent(r); ++i)
      check_operator_result(mapping, i, idxs...);
  }
}

template <class Extents, size_t PaddingValue, class... Args>
constexpr void test_operator_result(Args... args) {
  using Mapping = std::layout_right_padded<PaddingValue>::template mapping<Extents>;

  Mapping mapping(Extents(args...));
  check_operator_result(mapping);
}

template <class Mapping>
constexpr void test_operator_result(Mapping mapping) {
  check_operator_result(mapping);
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  {
    using Rank0Mapping = std::layout_right_padded<4>::mapping<std::extents<int>>;
    using Rank1Mapping = std::layout_right_padded<D>::mapping<std::extents<int, D>>;

    static_assert(operator_constraints<Rank0Mapping>);
    static_assert(!operator_constraints<Rank0Mapping, int>);

    static_assert(operator_constraints<Rank1Mapping, int>);
    static_assert(!operator_constraints<Rank1Mapping>);
    static_assert(!operator_constraints<Rank1Mapping, int, int>);
  }

  {
    using SignedMapping   = std::layout_right_padded<D>::mapping<std::extents<int, D>>;
    using UnsignedMapping = std::layout_right_padded<D>::mapping<std::extents<unsigned, D>>;

    static_assert(operator_constraints<SignedMapping, IntType>);
    static_assert(!operator_constraints<UnsignedMapping, IntType>);
  }

  {
    using SignedMapping   = std::layout_right_padded<D>::mapping<std::extents<signed char, D>>;
    using UnsignedMapping = std::layout_right_padded<D>::mapping<std::extents<unsigned char, D>>;

    static_assert(operator_constraints<SignedMapping, IntType>);
    static_assert(!operator_constraints<UnsignedMapping, IntType>);
  }

  test_operator_result<std::extents<int>, 4>();
  test_operator_result<std::extents<unsigned, 7>, 4>();
  test_operator_result<std::extents<unsigned, 5, 7>, 4>();
  test_operator_result<std::extents<signed char, D, 2, 3>, 4>(3);

  test_operator_result<std::extents<int>, D>();
  test_operator_result<std::extents<unsigned, D>, D>(7);
  test_operator_result(
      std::layout_right_padded<D>::mapping<std::extents<unsigned, 5, 7>>(std::extents<unsigned, 5, 7>(), 6));
  test_operator_result(
      std::layout_right_padded<D>::mapping<std::extents<unsigned, D, 2, 3>>(std::extents<unsigned, D, 2, 3>(3), 4));

  {
    using Mapping = std::layout_right_padded<4>::mapping<std::extents<int, 2, 2>>;

    static_assert(operator_constraints<Mapping, RValueInt, RValueInt>);

    constexpr Mapping mapping(std::extents<int, 2, 2>{});
    static_assert(mapping(RValueInt{1}, RValueInt{0}) == expected_result(mapping, RValueInt{1}, RValueInt{0}));
  }

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
