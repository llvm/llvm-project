//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template <class _Range>
// stride_view(_Range&&, range_difference_t<_Range>) -> stride_view<views::all_t<_Range>>;

#include "types.h"
#include <concepts>
#include <ranges>

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct Range {
  int* begin() const;
  int* end() const;
};

constexpr bool testCTAD() {
  int a[] = {1, 2, 3, 4, 5};

  using BaseRange = BasicTestRange<cpp17_input_iterator<int*>>;
  using BaseView  = BasicTestView<int*>;

  auto base_view  = BaseView(a, a + 5);
  auto base_range = BaseRange(cpp17_input_iterator<int*>(a), cpp17_input_iterator<int*>(a + 5));

  auto copied_stride_base_view = std::ranges::stride_view(base_view, 2);
  auto moved_stride_base_view  = std::ranges::stride_view(std::move(base_view), 2);

  auto copied_stride_base_range = std::ranges::stride_view(base_range, 2);
  auto moved_stride_base_range  = std::ranges::stride_view(std::move(base_range), 2);

  static_assert(std::same_as< decltype(copied_stride_base_view), std::ranges::stride_view<BaseView>>);
  static_assert(std::same_as< decltype(moved_stride_base_view), std::ranges::stride_view<BaseView>>);

  static_assert(
      std::same_as< decltype(copied_stride_base_range), std::ranges::stride_view<std::ranges::ref_view<BaseRange>> >);
  static_assert(
      std::same_as< decltype(moved_stride_base_range), std::ranges::stride_view<std::ranges::owning_view<BaseRange>> >);

  assert(*copied_stride_base_range.begin() == 1);
  assert(*moved_stride_base_range.begin() == 1);

  assert(*copied_stride_base_view.begin() == 1);
  assert(*moved_stride_base_view.begin() == 1);

  auto copied_stride_range_it = copied_stride_base_range.begin();
  copied_stride_range_it++;
  assert(*copied_stride_range_it == 3);
  copied_stride_range_it++;
  copied_stride_range_it++;
  assert(copied_stride_range_it == copied_stride_base_range.end());

  auto moved_stride_range_it = moved_stride_base_range.begin();
  moved_stride_range_it++;
  moved_stride_range_it++;
  assert(*moved_stride_range_it == 5);
  moved_stride_range_it++;
  assert(moved_stride_range_it == moved_stride_base_range.end());

  auto copied_stride_view_it = copied_stride_base_view.begin();
  copied_stride_view_it++;
  assert(*copied_stride_view_it == 3);
  copied_stride_view_it++;
  copied_stride_view_it++;
  assert(copied_stride_view_it == copied_stride_base_view.end());

  auto moved_stride_view_it = copied_stride_base_view.begin();
  moved_stride_view_it++;
  moved_stride_view_it++;
  assert(*moved_stride_view_it == 5);
  moved_stride_view_it++;
  assert(moved_stride_view_it == moved_stride_base_view.end());

  return true;
}

int main(int, char**) {
  testCTAD();
  static_assert(testCTAD());

  return 0;
}
