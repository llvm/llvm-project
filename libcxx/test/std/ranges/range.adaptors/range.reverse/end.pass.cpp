//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr reverse_iterator<iterator_t<V>> end();
// constexpr auto end() const requires common_range<const V>;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "types.h"

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Common bidirectional range.
  {
    auto rev = std::ranges::reverse_view(BidirRange{buffer});
    assert(rev.end().base().base() == buffer);
    assert(std::move(rev).end().base().base() == buffer);

    ASSERT_SAME_TYPE(decltype(rev.end()), std::reverse_iterator<bidirectional_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(std::move(rev).end()), std::reverse_iterator<bidirectional_iterator<int*>>);
  }
  // Const common bidirectional range.
  {
    const auto rev = std::ranges::reverse_view(BidirRange{buffer});
    assert(rev.end().base().base() == buffer);
    assert(std::move(rev).end().base().base() == buffer);

    ASSERT_SAME_TYPE(decltype(rev.end()), std::reverse_iterator<bidirectional_iterator<const int*>>);
    ASSERT_SAME_TYPE(decltype(std::move(rev).end()), std::reverse_iterator<bidirectional_iterator<const int*>>);
  }
  // Non-common, non-const (move only) bidirectional range.
  {
    auto rev = std::ranges::reverse_view(BidirSentRange<MoveOnly>{buffer});
    assert(std::move(rev).end().base().base() == buffer);

    ASSERT_SAME_TYPE(decltype(std::move(rev).end()), std::reverse_iterator<bidirectional_iterator<int*>>);
  }
  // Non-common, const bidirectional range.
  {
    auto rev = std::ranges::reverse_view(BidirSentRange<Copyable>{buffer});
    assert(rev.end().base().base() == buffer);
    assert(std::move(rev).end().base().base() == buffer);

    ASSERT_SAME_TYPE(decltype(rev.end()), std::reverse_iterator<bidirectional_iterator<int*>>);
    ASSERT_SAME_TYPE(decltype(std::move(rev).end()), std::reverse_iterator<bidirectional_iterator<int*>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
