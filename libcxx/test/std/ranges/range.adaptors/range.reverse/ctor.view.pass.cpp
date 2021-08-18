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

// constexpr explicit reverse_view(V r);

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "types.h"

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    BidirRange r{buffer};
    std::ranges::reverse_view<BidirRange> rev(r);
    assert(rev.base().ptr_ == buffer);
  }
  {
    const BidirRange r{buffer};
    const std::ranges::reverse_view<BidirRange> rev(r);
    assert(rev.base().ptr_ == buffer);
  }
  {
    std::ranges::reverse_view<BidirSentRange<MoveOnly>> rev(BidirSentRange<MoveOnly>{buffer});
    assert(std::move(rev).base().ptr_ == buffer);
  }
  {
    const std::ranges::reverse_view<BidirSentRange<Copyable>> rev(BidirSentRange<Copyable>{buffer});
    assert(rev.base().ptr_ == buffer);
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
