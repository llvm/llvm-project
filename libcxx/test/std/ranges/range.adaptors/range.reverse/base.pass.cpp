//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr V base() const& requires copy_constructible<V> { return base_; }
// constexpr V base() && { return std::move(base_); }

#include <ranges>

#include <cassert>
#include <concepts>

#include "types.h"

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test common ranges.
  {
    // Test non-const.
    {
      auto rev = std::ranges::reverse_view(BidirRange{buffer, buffer + 8});

      std::same_as<BidirRange> auto base = rev.base();
      assert(base.begin_ == buffer);
      assert(base.end_ == buffer + 8);

      std::same_as<BidirRange> auto moved = std::move(rev).base();
      assert(moved.begin_ == buffer);
      assert(moved.end_ == buffer + 8);
    }
    // Test const.
    {
      const auto rev = std::ranges::reverse_view(BidirRange{buffer, buffer + 8});

      std::same_as<BidirRange> auto base = rev.base();
      assert(base.begin_ == buffer);
      assert(base.end_ == buffer + 8);

      std::same_as<BidirRange> auto moved = std::move(rev).base();
      assert(moved.begin_ == buffer);
      assert(moved.end_ == buffer + 8);
    }
  }
  // Test non-common ranges.
  {
    // Test non-const (also move only).
    {
      auto rev = std::ranges::reverse_view(BidirSentRange<MoveOnly>{buffer, buffer + 8});
      std::same_as<BidirSentRange<MoveOnly>> auto base = std::move(rev).base();
      assert(base.begin_ == buffer);
      assert(base.end_ == buffer + 8);
    }
    // Test const.
    {
      const auto rev = std::ranges::reverse_view(BidirSentRange<Copyable>{buffer, buffer + 8});

      std::same_as<BidirSentRange<Copyable>> auto base = rev.base();
      assert(base.begin_ == buffer);
      assert(base.end_ == buffer + 8);

      std::same_as<BidirSentRange<Copyable>> auto moved = std::move(rev).base();
      assert(moved.begin_ == buffer);
      assert(moved.end_ == buffer + 8);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
