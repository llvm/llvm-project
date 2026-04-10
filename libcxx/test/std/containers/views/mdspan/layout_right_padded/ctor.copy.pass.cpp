//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// constexpr mapping(const mapping&) noexcept = default;
// constexpr mapping& operator=(const mapping&) noexcept = default;

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <utility>

#include "test_macros.h"

template <class Mapping>
constexpr void test_copy_semantics(const Mapping& source) {
  ASSERT_NOEXCEPT(Mapping(source));
  static_assert(noexcept(std::declval<Mapping&>() = source));

  Mapping copy(source);
  assert(copy == source);

  Mapping assigned;
  assigned = source;
  assert(assigned == source);
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  test_copy_semantics(std::layout_right_padded<4>::mapping<std::extents<int32_t>>());
  test_copy_semantics(
      std::layout_right_padded<4>::mapping<std::extents<uint32_t, 5, 7>>(std::extents<int32_t, 5, 7>()));
  test_copy_semantics(std::layout_right_padded<4>::mapping<std::extents<int8_t, D>>(std::extents<int8_t, D>(5)));
  test_copy_semantics(
      std::layout_right_padded<4>::mapping<std::extents<uint8_t, D, 7>>(std::extents<uint8_t, D, 7>(5)));

  test_copy_semantics(std::layout_right_padded<D>::mapping<std::extents<int32_t>>());
  test_copy_semantics(
      std::layout_right_padded<D>::mapping<std::extents<uint32_t, 5, 7>>(std::extents<int32_t, 5, 7>()));
  test_copy_semantics(std::layout_right_padded<D>::mapping<std::extents<int8_t, D>>(std::extents<int8_t, D>(5), 5));
  test_copy_semantics(
      std::layout_right_padded<D>::mapping<std::extents<uint8_t, D, 7>>(std::extents<uint8_t, D, 7>(5)));

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
