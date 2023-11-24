//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS: -Wno-ctad-maybe-unsupported

// <mdspan>

#include <mdspan>
#include <type_traits>
#include <concepts>
#include <cassert>

#include "test_macros.h"

// mdspan

// layout_stride::mapping does not have explicit deduction guides,
// but implicit deduction guides for constructor taking extents and strides
// should work

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  ASSERT_SAME_TYPE(decltype(std::layout_stride::mapping(std::extents<int>(), std::array<unsigned, 0>())),
                   std::layout_stride::template mapping<std::extents<int>>);
  ASSERT_SAME_TYPE(decltype(std::layout_stride::mapping(std::extents<int, 4>(), std::array<char, 1>{1})),
                   std::layout_stride::template mapping<std::extents<int, 4>>);
  ASSERT_SAME_TYPE(decltype(std::layout_stride::mapping(std::extents<int, D>(), std::array<char, 1>{1})),
                   std::layout_stride::template mapping<std::extents<int, D>>);
  ASSERT_SAME_TYPE(
      decltype(std::layout_stride::mapping(std::extents<unsigned, D, 3>(), std::array<int64_t, 2>{3, 100})),
      std::layout_stride::template mapping<std::extents<unsigned, D, 3>>);

  ASSERT_SAME_TYPE(decltype(std::layout_stride::mapping(std::extents<int>(), std::span<unsigned, 0>())),
                   std::layout_stride::template mapping<std::extents<int>>);
  ASSERT_SAME_TYPE(decltype(std::layout_stride::mapping(std::extents<int, 4>(), std::declval<std::span<char, 1>>())),
                   std::layout_stride::template mapping<std::extents<int, 4>>);
  ASSERT_SAME_TYPE(decltype(std::layout_stride::mapping(std::extents<int, D>(), std::declval<std::span<char, 1>>())),
                   std::layout_stride::template mapping<std::extents<int, D>>);
  ASSERT_SAME_TYPE(
      decltype(std::layout_stride::mapping(std::extents<unsigned, D, 3>(), std::declval<std::span<int64_t, 2>>())),
      std::layout_stride::template mapping<std::extents<unsigned, D, 3>>);
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
