//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr explicit reverse_view(V r);

#include <ranges>

#include <cassert>
#include <type_traits>
#include <utility>

#include "types.h"

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    BidirRange r{buffer, buffer + 8};
    std::ranges::reverse_view<BidirRange> rev(r);
    assert(rev.base().begin_ == buffer);
    assert(rev.base().end_ == buffer + 8);
  }
  {
    const BidirRange r{buffer, buffer + 8};
    const std::ranges::reverse_view<BidirRange> rev(r);
    assert(rev.base().begin_ == buffer);
    assert(rev.base().end_ == buffer + 8);
  }
  {
    std::ranges::reverse_view<BidirSentRange<MoveOnly>> rev(BidirSentRange<MoveOnly>{buffer, buffer + 8});
    auto moved = std::move(rev).base();
    assert(moved.begin_ == buffer);
    assert(moved.end_ == buffer + 8);
  }
  {
    const std::ranges::reverse_view<BidirSentRange<Copyable>> rev(BidirSentRange<Copyable>{buffer, buffer + 8});
    assert(rev.base().begin_ == buffer);
    assert(rev.base().end_ == buffer + 8);
  }
  {
    // Make sure this ctor is marked as "explicit".
    static_assert( std::is_constructible_v<std::ranges::reverse_view<BidirRange>, BidirRange>);
    static_assert(!std::is_convertible_v<std::ranges::reverse_view<BidirRange>, BidirRange>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
