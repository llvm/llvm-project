//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// constexpr mapping(const extents_type& e, index_type padding_stride);

#include <array>
#include <cassert>
#include <cstddef>
#include <mdspan>

#include "../ConvertibleToIntegral.h"

template <class Extents, class Padding>
constexpr void test_construction(
    Extents extents, Padding padding, std::array<typename Extents::index_type, Extents::rank()> expected_strides) {
  using Mapping = std::layout_right_padded<std::dynamic_extent>::mapping<Extents>;
  static_assert(Mapping::padding_value == std::dynamic_extent);

  Mapping mapping(extents, padding);
  assert(mapping.extents() == extents);

  for (typename Mapping::rank_type r = 0; r < Mapping::extents_type::rank(); ++r)
    assert(mapping.stride(r) == expected_strides[r]);
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  // clang-format off
  test_construction(std::extents<unsigned>(), 3,                   {});
  test_construction(std::extents<unsigned, D>(7), 1337,            {1});
  test_construction(std::extents<unsigned, D, 7>(5), 6,            {12, 1});
  test_construction(std::extents<unsigned, 5, 7>(), 4,             {8, 1});
  test_construction(std::extents<unsigned, D, 7, D>(7, 3), 4,      {28, 4, 1});
  test_construction(std::extents<unsigned, 0, 7>(), 4,             {8, 1});

  test_construction(std::extents<unsigned, 5, 7>(), RValueInt{4},  {8, 1});
  // clang-format on

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
