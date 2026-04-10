//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// namespace std {
//   template<class Extents>
//   class layout_right_padded::mapping {
//     ...
//     static constexpr bool is_always_unique() noexcept;
//     static constexpr bool is_always_exhaustive() noexcept;
//     static constexpr bool is_always_strided() noexcept;
//     static constexpr bool is_unique() noexcept;
//     constexpr bool is_exhaustive() const noexcept;
//     static constexpr bool is_strided() noexcept;
//   };
// }

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <utility>

#include "test_macros.h"

template <class Mapping>
constexpr void test_properties(const Mapping& mapping, bool exhaustive, bool always_exhaustive) {
  const Mapping const_mapping = mapping;

  assert(Mapping::is_unique());
  assert(mapping.is_unique());
  assert(const_mapping.is_unique());

  assert(mapping.is_exhaustive() == exhaustive);
  assert(const_mapping.is_exhaustive() == exhaustive);

  assert(Mapping::is_strided());
  assert(mapping.is_strided());
  assert(const_mapping.is_strided());

  assert(Mapping::is_always_unique());
  assert(Mapping::is_always_exhaustive() == always_exhaustive);
  assert(Mapping::is_always_strided());

  ASSERT_NOEXCEPT(std::declval<Mapping>().is_unique());
  ASSERT_NOEXCEPT(std::declval<Mapping>().is_exhaustive());
  ASSERT_NOEXCEPT(std::declval<Mapping>().is_strided());
  ASSERT_NOEXCEPT(Mapping::is_always_unique());
  ASSERT_NOEXCEPT(Mapping::is_always_exhaustive());
  ASSERT_NOEXCEPT(Mapping::is_always_strided());
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  // clang-format off
  test_properties(std::layout_right_padded<4>::mapping<std::extents<int32_t>>(),           true,   true);
  test_properties(std::layout_right_padded<4>::mapping<std::extents<int32_t,  5>>(),       true,   true);
  test_properties(std::layout_right_padded<4>::mapping<std::extents<uint32_t, 3, 4>>(),    true,   true);
  test_properties(std::layout_right_padded<4>::mapping<std::extents<uint32_t, 4, 6>>(),    false,  false);
  test_properties(std::layout_right_padded<4>::mapping<std::extents<int32_t,  6, D>>(
                                                       std::extents<int32_t,  6, D>(7)),   false,  false);
  test_properties(std::layout_right_padded<4>::mapping<std::extents<int32_t,  6, D>>(
                                                       std::extents<int32_t,  6, D>(8)),   true,   false);
  test_properties(std::layout_right_padded<4>::mapping<std::extents<uint32_t>>(),          true,   true);
  test_properties(std::layout_right_padded<4>::mapping<std::extents<uint32_t, 0, 6>>(),    false,  false);

  test_properties(std::layout_right_padded<D>::mapping<std::extents<int32_t,  6, D>>(
                                                       std::extents<int32_t,  6, D>(4), 4), true,   false);
  test_properties(std::layout_right_padded<D>::mapping<std::extents<int32_t,  6, D>>(
                                                       std::extents<int32_t,  6, D>(6), 4), false,  false);
  // clang-format on

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
