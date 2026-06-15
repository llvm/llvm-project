//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// constexpr index_type required_span_size() const noexcept;

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mdspan>

#include "test_macros.h"

template <class Mapping>
constexpr void test_required_span_size(const Mapping& mapping, typename Mapping::index_type expected_size) {
  ASSERT_NOEXCEPT(mapping.required_span_size());
  assert(mapping.required_span_size() == expected_size);
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  // clang-format off
  test_required_span_size(std::layout_left_padded<4>::mapping<std::extents<int32_t>>(),                                             1);
  test_required_span_size(std::layout_left_padded<4>::mapping<std::extents<int32_t,  D>>(std::extents<int32_t,  D>(0)),             0);
  test_required_span_size(std::layout_left_padded<4>::mapping<std::extents<uint32_t, D>>(std::extents<uint32_t, D>(7)),             7);
  test_required_span_size(std::layout_left_padded<4>::mapping<std::extents<uint32_t, 5,  7>>(),                                     53);
  test_required_span_size(std::layout_left_padded<4>::mapping<std::extents<int64_t,  D,  2, 3>>(std::extents<int64_t, D, 2, 3>(7)), 47);
  test_required_span_size(std::layout_left_padded<4>::mapping<std::extents<int64_t,  15, 0>>(),                                     0);

  test_required_span_size(std::layout_left_padded<D>::mapping<std::extents<int32_t>>(),                                         1);
  test_required_span_size(std::layout_left_padded<D>::mapping<std::extents<int32_t,  D>>(std::extents<int32_t,  D>(0)),         0);
  test_required_span_size(std::layout_left_padded<D>::mapping<std::extents<uint32_t, D>>(std::extents<uint32_t, D>(7)),         7);
  test_required_span_size(std::layout_left_padded<D>::mapping<std::extents<uint32_t, 5, 7>>(),                                  35);
  test_required_span_size(std::layout_left_padded<D>::mapping<std::extents<int64_t,  5, 7>>(std::extents<int64_t, 5, 7>(), 6),  41);
  test_required_span_size(std::layout_left_padded<D>::mapping<std::extents<int64_t,  0, 7>>(),                                  0);
  // clang-format on

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
