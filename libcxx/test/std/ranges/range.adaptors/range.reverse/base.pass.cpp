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

// constexpr V base() const& requires copy_constructible<V> { return base_; }
// constexpr V base() && { return std::move(base_); }

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "types.h"

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Test common ranges.
  {
    // Test non-const.
    {
      auto rev = std::ranges::reverse_view(BidirRange{buffer});
      assert(rev.base().ptr_ == buffer);
      assert(std::move(rev).base().ptr_ == buffer);

      ASSERT_SAME_TYPE(decltype(rev.base()), BidirRange);
      ASSERT_SAME_TYPE(decltype(std::move(rev).base()), BidirRange);
    }
    // Test const.
    {
      const auto rev = std::ranges::reverse_view(BidirRange{buffer});
      assert(rev.base().ptr_ == buffer);
      assert(std::move(rev).base().ptr_ == buffer);

      ASSERT_SAME_TYPE(decltype(rev.base()), BidirRange);
      ASSERT_SAME_TYPE(decltype(std::move(rev).base()), BidirRange);
    }
  }
  // Test non-common ranges.
  {
    // Test non-const (also move only).
    {
      auto rev = std::ranges::reverse_view(BidirSentRange<MoveOnly>{buffer});
      assert(std::move(rev).base().ptr_ == buffer);

      ASSERT_SAME_TYPE(decltype(std::move(rev).base()), BidirSentRange<MoveOnly>);
    }
    // Test const.
    {
      const auto rev = std::ranges::reverse_view(BidirSentRange<Copyable>{buffer});
      assert(rev.base().ptr_ == buffer);
      assert(std::move(rev).base().ptr_ == buffer);

      ASSERT_SAME_TYPE(decltype(rev.base()), BidirSentRange<Copyable>);
      ASSERT_SAME_TYPE(decltype(std::move(rev).base()), BidirSentRange<Copyable>);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
