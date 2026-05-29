//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// constexpr mapping() noexcept;

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mdspan>

#include "test_macros.h"

template <class Mapping>
constexpr void test_construction() {
  using Extents = Mapping::extents_type;

  ASSERT_NOEXCEPT(Mapping{});
  Mapping mapping;
  Mapping extents_mapping{Extents()};

  assert(mapping == extents_mapping);
  assert(mapping.extents() == Extents());
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  test_construction<std::layout_right_padded<4>::mapping<std::extents<int32_t>>>();
  test_construction<std::layout_right_padded<4>::mapping<std::extents<int32_t, 1>>>();
  test_construction<std::layout_right_padded<4>::mapping<std::extents<uint32_t, 2, 3>>>();
  test_construction<std::layout_right_padded<4>::mapping<std::extents<uint32_t, 4, 5, D>>>();

  test_construction<std::layout_right_padded<D>::mapping<std::extents<int32_t>>>();
  test_construction<std::layout_right_padded<D>::mapping<std::extents<int32_t, 1>>>();
  test_construction<std::layout_right_padded<D>::mapping<std::extents<uint32_t, 2, 3>>>();
  test_construction<std::layout_right_padded<D>::mapping<std::extents<uint32_t, 4, 5, D>>>();

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
