//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// Test that stride_view's iterator member functions are properly marked nodiscard.

#include <ranges>
#include <utility>

#include "../../../../../std/ranges/range.adaptors/range.stride.view/types.h"

void test_base_nodiscard() {
  {
    int range[] = {1, 2, 3};
    auto view   = std::ranges::views::stride(range, 3);
    auto it     = view.begin();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it.base();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::move(it).base();
  }
}

void test_dereference_nodiscard() {
  {
    int range[] = {1, 2, 3};
    auto view   = std::ranges::views::stride(range, 3);
    auto it     = view.begin();
    ++it;
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *std::as_const(it);
  }
}

constexpr bool test_iter_move_nodiscard() {
  {
    int a[] = {4, 3, 2, 1};

    int iter_move_counter(0);
    using View       = IterMoveIterSwapTestRange<int*, true, true>;
    using StrideView = std::ranges::stride_view<View>;
    auto svb         = StrideView(View(a, a + 4, &iter_move_counter), 1).begin();

    static_assert(std::is_same_v<int, decltype(std::ranges::iter_move(svb))>);
    static_assert(noexcept(std::ranges::iter_move(svb)));

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::ranges::iter_move(svb);
  }
  return true;
}

constexpr bool test_non_forward_operator_minus_nodiscard() {
  int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  using Base = BasicTestView<SizedInputIter, SizedInputIter>;

  auto base_view_offset_zero             = Base(SizedInputIter(arr), SizedInputIter(arr + 10));
  auto base_view_offset_one              = Base(SizedInputIter(arr + 1), SizedInputIter(arr + 10));
  auto stride_view_over_base_zero_offset = std::ranges::stride_view(base_view_offset_zero, 3);
  auto stride_view_over_base_one_offset  = std::ranges::stride_view(base_view_offset_one, 3);

  auto sv_zero_offset_begin = stride_view_over_base_zero_offset.begin();
  auto sv_one_offset_begin  = stride_view_over_base_one_offset.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv_one_offset_begin - sv_zero_offset_begin;
  return true;
}

constexpr bool test_forward_operator_minus_nodiscard() {
  int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  using Base = BasicTestView<int*, int*>;

  auto base_view_offset_zero             = Base(arr, arr + 10);
  auto base_view_offset_one              = Base(arr + 1, arr + 10);
  auto stride_view_over_base_zero_offset = std::ranges::stride_view(base_view_offset_zero, 3);
  auto stride_view_over_base_one_offset  = std::ranges::stride_view(base_view_offset_one, 3);

  auto sv_zero_offset_begin = stride_view_over_base_zero_offset.begin();
  auto sv_one_offset_begin  = stride_view_over_base_one_offset.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv_zero_offset_begin + 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  1 + sv_zero_offset_begin;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv_one_offset_begin - 1;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv_one_offset_begin - sv_zero_offset_begin;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::default_sentinel_t() - sv_zero_offset_begin;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv_zero_offset_begin - std::default_sentinel_t();

  return true;
}

void test_subscript_nodiscard() {
  {
    int range[] = {1, 2, 3};
    auto view   = std::ranges::views::stride(range, 3);
    auto it     = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[0];
  }
}
