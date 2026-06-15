//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// constexpr index_type stride(rank_type) const noexcept;

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mdspan>

#include "test_macros.h"

template <typename Mapping>
constexpr void test_stride(const Mapping& mapping,
                           std::array<typename Mapping::index_type, Mapping::extents_type::rank()> expected_strides) {
  ASSERT_NOEXCEPT(mapping.stride(0));
  for (typename Mapping::rank_type r = 0; r < Mapping::extents_type::rank(); ++r)
    assert(mapping.stride(r) == expected_strides[r]);
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  // clang-format off
  test_stride(std::layout_left_padded<4>::mapping<std::extents<int32_t,  D>>(
                                                  std::extents<int32_t,  D>(7)),                    {1});
  test_stride(std::layout_left_padded<4>::mapping<std::extents<int32_t,  7>>(),                     {1});
  test_stride(std::layout_left_padded<4>::mapping<std::extents<uint32_t, 7, 8>>(),                  {1, 8});
  test_stride(std::layout_left_padded<4>::mapping<std::extents<uint32_t, D, 8, D, D>>(
                                                  std::extents<uint32_t, D, 8, D, D>(7, 9, 10)),    {1, 8, 64, 576});

  test_stride(std::layout_left_padded<D>::mapping<std::extents<int32_t,  D>>(
                                                  std::extents<int32_t,  D>(7)),                    {1});
  test_stride(std::layout_left_padded<D>::mapping<std::extents<int32_t,  7>>(),                     {1});
  test_stride(std::layout_left_padded<D>::mapping<std::extents<uint32_t, 7, 8>>(),                  {1, 7});
  test_stride(std::layout_left_padded<D>::mapping<std::extents<uint32_t, D, 8, D, D>>(
                                                  std::extents<uint32_t, D, 8, D, D>(7, 9, 10)),    {1, 7, 56, 504});
  test_stride(std::layout_left_padded<D>::mapping<std::extents<int32_t,  7, 8>>(
                                                  std::extents<int32_t,  7, 8>(), 6),               {1, 12});
  test_stride(std::layout_left_padded<D>::mapping<std::extents<uint32_t, D, 8, D, D>>(
                                                  std::extents<uint32_t, D, 8, D, D>(7, 9, 10), 6), {1, 12, 96, 864});
  // clang-format on

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
