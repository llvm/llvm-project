//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template <class R>
// stride_view(R&&, range_difference_t<R>) -> stride_view<views::all_t<R>>;

#include <cassert>
#include <concepts>
#include <ranges>
#include <utility>

#include "types.h"

struct View : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct Range {
  int* begin() const;
  int* end() const;
};

constexpr bool test() {
  int a[] = {1, 2, 3, 4, 5};

  using BaseRange = BasicTestRange<cpp17_input_iterator<int*>>;
  using BaseView  = BasicTestView<int*>;

  auto base_view      = BaseView(a, a + 5);
  auto base_view_move = BaseView(a, a + 5);

  auto base_range      = BaseRange(cpp17_input_iterator<int*>(a), cpp17_input_iterator<int*>(a + 5));
  auto base_range_move = BaseRange(cpp17_input_iterator<int*>(a), cpp17_input_iterator<int*>(a + 5));

  // Deduction from lvalue view and rvalue view.
  auto sv_view      = std::ranges::stride_view(base_view, 2);
  auto sv_view_move = std::ranges::stride_view(std::move(base_view_move), 2);

  // Deduction from lvalue range (-> ref_view) and rvalue range (-> owning_view).
  auto sv_range      = std::ranges::stride_view(base_range, 2);
  auto sv_range_move = std::ranges::stride_view(std::move(base_range_move), 2);

  // Verify deduced types for views.
  static_assert(std::same_as<decltype(sv_view), std::ranges::stride_view<BaseView>>);
  static_assert(std::same_as<decltype(sv_view_move), std::ranges::stride_view<BaseView>>);

  // Verify deduced types for ranges: lvalue -> ref_view, rvalue -> owning_view.
  static_assert(std::same_as<decltype(sv_range), std::ranges::stride_view<std::ranges::ref_view<BaseRange>> >);
  static_assert(std::same_as<decltype(sv_range_move), std::ranges::stride_view<std::ranges::owning_view<BaseRange>> >);

  // Verify begin() produces the first element.
  assert(*sv_range.begin() == 1);
  assert(*sv_range_move.begin() == 1);
  assert(*sv_view.begin() == 1);
  assert(*sv_view_move.begin() == 1);

  // Verify iteration with stride 2 over a range.
  auto it = sv_range.begin();
  it++;
  assert(*it == 3);
  it++;
  it++;
  assert(it == sv_range.end());

  auto it2 = sv_range_move.begin();
  it2++;
  it2++;
  assert(*it2 == 5);
  it2++;
  assert(it2 == sv_range_move.end());

  // Verify iteration with stride 2 over a view.
  auto it3 = sv_view.begin();
  it3++;
  assert(*it3 == 3);
  it3++;
  it3++;
  assert(it3 == sv_view.end());

  auto it4 = sv_view.begin();
  it4++;
  it4++;
  assert(*it4 == 5);
  it4++;
  assert(it4 == sv_view_move.end());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
