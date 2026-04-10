//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <mdspan>

// template<class LayoutRightPaddedMapping>
// friend constexpr bool operator==(const mapping&, const LayoutRightPaddedMapping&) noexcept;

#include <cassert>
#include <cstddef>
#include <mdspan>

#include "test_macros.h"

template <class LeftMapping, class RightMapping>
constexpr void test_comparison(bool equal, const LeftMapping& left, const RightMapping& right) {
  ASSERT_NOEXCEPT(left == right);
  assert((left == right) == equal);
  assert((left != right) == !equal);
}

struct DoesNotMatch {
  constexpr bool does_not_match() const { return true; }
};

constexpr DoesNotMatch compare_layout_mappings(...) { return {}; }

template <class LeftMapping, class RightMapping>
constexpr auto compare_layout_mappings(const LeftMapping& left, const RightMapping& right) -> decltype(left == right) {
  (void)left;
  (void)right;
  return true;
}

constexpr bool test() {
  constexpr size_t D = std::dynamic_extent;

  {
    using rank0_left  = std::layout_right_padded<0>::mapping<std::extents<size_t>>;
    using rank0_right = std::layout_right_padded<4>::mapping<std::extents<size_t>>;
    test_comparison(true, rank0_left(), rank0_right());
  }

  {
    using rank1_fixed   = std::layout_right_padded<4>::mapping<std::extents<size_t, D>>;
    using rank1_dynamic = std::layout_right_padded<D>::mapping<std::extents<size_t, 5>>;
    test_comparison(true, rank1_fixed(std::extents<size_t, D>(5)), rank1_dynamic(std::extents<size_t, 5>(), 99));
    test_comparison(false, rank1_fixed(std::extents<size_t, D>(3)), rank1_dynamic(std::extents<size_t, 5>(), 99));
  }

  {
    using fixed_mapping   = std::layout_right_padded<4>::mapping<std::extents<size_t, 3, 7>>;
    using dynamic_mapping = std::layout_right_padded<D>::mapping<std::extents<size_t, 3, 7>>;
    test_comparison(true, fixed_mapping(), dynamic_mapping(std::extents<size_t, 3, 7>(), 4));
    test_comparison(false, fixed_mapping(), dynamic_mapping(std::extents<size_t, 3, 7>(), 6));
    test_comparison(false, fixed_mapping(), std::layout_right_padded<4>::mapping<std::extents<size_t, 3, 4>>());
  }

  {
    using dynamic_mapping = std::layout_right_padded<D>::mapping<std::extents<size_t, D, 3>>;
    test_comparison(
        true, dynamic_mapping(std::extents<size_t, D, 3>(5), 5), dynamic_mapping(std::extents<size_t, D, 3>(5), 5));
    test_comparison(
        false, dynamic_mapping(std::extents<size_t, D, 3>(5), 5), dynamic_mapping(std::extents<size_t, D, 3>(5), 8));
  }

  static_assert(compare_layout_mappings(
      std::layout_right_padded<4>::mapping<std::extents<int, D>>(std::extents<int, D>(5)),
      std::layout_right_padded<4>::mapping<std::extents<int, 5>>()));
  static_assert(compare_layout_mappings(std::layout_right_padded<4>::mapping<std::extents<int>>(),
                                        std::layout_right_padded<4>::mapping<std::extents<int, 1>>())
                    .does_not_match());
  static_assert(compare_layout_mappings(std::layout_right_padded<4>::mapping<std::extents<int, 5, 7>>(),
                                        std::layout_right_padded<4>::mapping<std::extents<int, 5>>())
                    .does_not_match());

  return true;
}

int main() {
  test();
  static_assert(test());
  return 0;
}
