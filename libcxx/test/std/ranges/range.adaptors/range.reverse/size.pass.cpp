//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto size() requires sized_range<V>;
// constexpr auto size() const requires sized_range<const V>;

#include <ranges>

#include <cassert>
#include <utility>

#include "test_macros.h"
#include "types.h"

// end -  begin = 8, but size may return something else.
template<CopyCategory CC>
struct BidirSizedRange : std::ranges::view_base {
  int *ptr_;
  std::size_t size_;

  constexpr BidirSizedRange(int *ptr, std::size_t size) : ptr_(ptr), size_(size) {}
  constexpr BidirSizedRange(const BidirSizedRange &) requires (CC == Copyable) = default;
  constexpr BidirSizedRange(BidirSizedRange &&) requires (CC == MoveOnly) = default;
  constexpr BidirSizedRange& operator=(const BidirSizedRange &) requires (CC == Copyable) = default;
  constexpr BidirSizedRange& operator=(BidirSizedRange &&) requires (CC == MoveOnly) = default;

  constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>{ptr_}; }
  constexpr bidirectional_iterator<const int*> begin() const { return bidirectional_iterator<const int*>{ptr_}; }
  constexpr bidirectional_iterator<int*> end() { return bidirectional_iterator<int*>{ptr_ + 8}; }
  constexpr bidirectional_iterator<const int*> end() const { return bidirectional_iterator<const int*>{ptr_ + 8}; }

  constexpr std::size_t size() const { return size_; }
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Non-common, non-const bidirectional range.
  {
    auto rev = std::ranges::reverse_view(BidirSizedRange<Copyable>{buffer, 4});
    assert(std::ranges::size(rev) == 4);
    assert(rev.size() == 4);
    assert(std::move(rev).size() == 4);

    ASSERT_SAME_TYPE(decltype(rev.size()), std::size_t);
    ASSERT_SAME_TYPE(decltype(std::move(rev).size()), std::size_t);
  }
  // Non-common, const bidirectional range.
  {
    const auto rev = std::ranges::reverse_view(BidirSizedRange<Copyable>{buffer, 4});
    assert(std::ranges::size(rev) == 4);
    assert(rev.size() == 4);
    assert(std::move(rev).size() == 4);

    ASSERT_SAME_TYPE(decltype(rev.size()), std::size_t);
    ASSERT_SAME_TYPE(decltype(std::move(rev).size()), std::size_t);
  }
  // Non-common, non-const (move only) bidirectional range.
  {
    auto rev = std::ranges::reverse_view(BidirSizedRange<MoveOnly>{buffer, 4});
    assert(std::move(rev).size() == 4);

    ASSERT_SAME_TYPE(decltype(std::move(rev).size()), std::size_t);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
