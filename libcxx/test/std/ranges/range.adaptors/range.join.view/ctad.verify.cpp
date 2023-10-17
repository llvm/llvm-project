//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: !c++experimental

// template<class R>
//   explicit join_view(R&&) -> join_view<views::all_t<R>>;

// Tests that the deduction guide is explicit.

#include <ranges>

#include "test_iterators.h"

template<class T>
struct Range {
  friend T* begin(Range&) { return nullptr; }
  friend T* begin(Range const&) { return nullptr; }
  friend sentinel_wrapper<T*> end(Range&) { return sentinel_wrapper<T*>(nullptr); }
  friend sentinel_wrapper<T*> end(Range const&) { return sentinel_wrapper<T*>(nullptr); }
};

void testExplicitCTAD() {
  Range<Range<int>> r;
  std::ranges::join_view v = r; // expected-error {{no viable constructor or deduction guide for deduction of template arguments of 'join_view'}}
}
