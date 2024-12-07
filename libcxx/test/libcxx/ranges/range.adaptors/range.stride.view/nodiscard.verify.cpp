//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Test that
// std::view::stride()
// std::ranges::stride_view::base()
// std::ranges::stride_view::begin()
// std::ranges::stride_view::end()
// std::ranges::stride_view::size()
// std::ranges::stride_view::stride()
// are all marked nodiscard.

#include <ranges>

#include "../../../../std/ranges/range.adaptors/range.stride.view/types.h"

void test_base_nodiscard() {
  const std::vector<int> intv = {1, 2, 3};
  auto sv = std::ranges::stride_view(intv, 3);

  sv.base(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(sv).base(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void test_begin_nodiscard() {
  const auto const_sv = std::views::stride(SimpleCommonConstView{}, 2);
  auto unsimple_sv    = std::views::stride(UnsimpleConstView{}, 2);

  const_sv.begin();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  unsimple_sv.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}


void test_views_stride_nodiscard() {
  const int range[] = {1, 2, 3};

  std::views::stride( // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
      range,
      2);
}

void test_end_nodiscard() {
  const int range[] = {1, 2, 3};

  const auto const_sv = std::views::stride(range, 2);
  auto unsimple_sv    = std::views::stride(UnsimpleConstView{}, 2);

  const_sv.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  unsimple_sv.end(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void test_size_nodiscard() {
  auto sv             = std::views::stride(SimpleNoConstSizedCommonView(), 2);
  const auto const_sv = std::views::stride(SimpleCommonConstView(), 2);

  sv.size();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  const_sv.size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

void test_stride_nodiscard() {
  const int range[] = {1, 2, 3};
  auto const_sv = std::views::stride(range, 2);
  const_sv.stride(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
