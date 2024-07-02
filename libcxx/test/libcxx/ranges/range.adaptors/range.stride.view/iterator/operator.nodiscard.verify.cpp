//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr decltype(auto) operator[](difference_type __n) const

#include <ranges>

#include "../../../../../std/ranges/range.adaptors/range.stride.view/types.h"

constexpr bool test_non_forward_operator_minus() {
  int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  using Base = BasicTestView<SizedInputIterator, SizedInputIterator>;

  auto base_view_offset_zero             = Base(SizedInputIterator(arr), SizedInputIterator(arr + 10));
  auto base_view_offset_one              = Base(SizedInputIterator(arr + 1), SizedInputIterator(arr + 10));
  auto stride_view_over_base_zero_offset = std::ranges::stride_view(base_view_offset_zero, 3);
  auto stride_view_over_base_one_offset  = std::ranges::stride_view(base_view_offset_one, 3);

  auto sv_zero_offset_begin = stride_view_over_base_zero_offset.begin();
  auto sv_one_offset_begin  = stride_view_over_base_one_offset.begin();

  sv_one_offset_begin - // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
      sv_zero_offset_begin;
  return true;
}

constexpr bool test_forward_operator_minus() {
  // Test the forward-range operator- between two iterators (i.e., no ceil).
  using Base = BasicTestView<int*, int*>;
  int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  auto base_view_offset_zero             = Base(arr, arr + 10);
  auto base_view_offset_one              = Base(arr + 1, arr + 10);
  auto stride_view_over_base_zero_offset = std::ranges::stride_view(base_view_offset_zero, 3);
  auto stride_view_over_base_one_offset  = std::ranges::stride_view(base_view_offset_one, 3);

  auto sv_zero_offset_begin = stride_view_over_base_zero_offset.begin();
  auto sv_one_offset_begin  = stride_view_over_base_one_offset.begin();

  sv_zero_offset_begin + // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
      1;
  1 + sv_zero_offset_begin; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  sv_one_offset_begin - 1; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  sv_one_offset_begin - // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
      sv_zero_offset_begin;

  return true;
}
