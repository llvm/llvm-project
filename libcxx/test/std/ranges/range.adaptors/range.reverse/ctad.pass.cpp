//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class R>
//   reverse_view(R&&) -> reverse_view<views::all_t<R>>;

#include <ranges>

#include <concepts>
#include <utility>

#include "test_iterators.h"

struct View : std::ranges::view_base {
  friend int* begin(View&) { return nullptr; }
  friend int* begin(View const&) { return nullptr; }
  friend sentinel_wrapper<int*> end(View&) { return sentinel_wrapper<int*>(nullptr); }
  friend sentinel_wrapper<int*> end(View const&) { return sentinel_wrapper<int*>(nullptr); }
};

struct Range {
  friend int* begin(Range&) { return nullptr; }
  friend int* begin(Range const&) { return nullptr; }
  friend sentinel_wrapper<int*> end(Range&) { return sentinel_wrapper<int*>(nullptr); }
  friend sentinel_wrapper<int*> end(Range const&) { return sentinel_wrapper<int*>(nullptr); }
};

struct BorrowedRange {
  friend int* begin(BorrowedRange&) { return nullptr; }
  friend int* begin(BorrowedRange const&) { return nullptr; }
  friend sentinel_wrapper<int*> end(BorrowedRange&) { return sentinel_wrapper<int*>(nullptr); }
  friend sentinel_wrapper<int*> end(BorrowedRange const&) { return sentinel_wrapper<int*>(nullptr); }
};

template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedRange> = true;

int main(int, char**) {
  View v;
  Range r;
  BorrowedRange br;

  {
    std::same_as<std::ranges::reverse_view<View>> auto x = std::ranges::reverse_view(v);
    (void)x;
  }
  {
    std::same_as<std::ranges::reverse_view<std::ranges::ref_view<Range>>> auto x = std::ranges::reverse_view(r);
    (void)x;
    // std::ranges::reverse_view(std::move(r)) is invalid. RValue range must be borrowed.
  }
  {
    std::same_as<std::ranges::reverse_view<std::ranges::ref_view<BorrowedRange>>> auto x = std::ranges::reverse_view(br);
    (void)x;
  }
  {
    using Subrange = std::ranges::subrange<int *, sentinel_wrapper<int *>, std::ranges::subrange_kind::unsized>;
    std::same_as<std::ranges::reverse_view<Subrange>> auto x = std::ranges::reverse_view(std::move(br));
    (void)x;
  }

  return 0;
}
