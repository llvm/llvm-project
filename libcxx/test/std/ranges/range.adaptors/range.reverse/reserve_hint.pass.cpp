//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr auto reserve_hint()
//     requires approximately_sized_range<V>
// constexpr auto reserve_hint() const
//     requires approximately_sized_range<const V>

#include <cassert>
#include <cstddef>
#include <ranges>
#include <utility>

#include "test_macros.h"
#include "types.h"

// end -  begin = 8, but size may return something else.
template <CopyCategory CC>
struct BidirApproxSizedRange : std::ranges::view_base {
  int* ptr_;
  std::size_t reserve_hint_;

  constexpr BidirApproxSizedRange(int* ptr, std::size_t reserve_hint) : ptr_(ptr), reserve_hint_(reserve_hint) {}
  constexpr BidirApproxSizedRange(const BidirApproxSizedRange&)
    requires(CC == Copyable)
  = default;
  constexpr BidirApproxSizedRange(BidirApproxSizedRange&&)
    requires(CC == MoveOnly)
  = default;
  constexpr BidirApproxSizedRange& operator=(const BidirApproxSizedRange&)
    requires(CC == Copyable)
  = default;
  constexpr BidirApproxSizedRange& operator=(BidirApproxSizedRange&&)
    requires(CC == MoveOnly)
  = default;

  constexpr bidirectional_iterator<int*> begin() { return bidirectional_iterator<int*>{ptr_}; }
  constexpr bidirectional_iterator<const int*> begin() const { return bidirectional_iterator<const int*>{ptr_}; }
  constexpr bidirectional_iterator<int*> end() { return bidirectional_iterator<int*>{ptr_ + 8}; }
  constexpr bidirectional_iterator<const int*> end() const { return bidirectional_iterator<const int*>{ptr_ + 8}; }

  constexpr std::size_t size() const { return reserve_hint_; }
};

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Non-common, non-const bidirectional range.
  {
    auto rev = std::ranges::reverse_view(BidirApproxSizedRange<Copyable>{buffer, 4});
    assert(std::ranges::reserve_hint(rev) == 4);
    assert(rev.reserve_hint() == 4);
    assert(std::move(rev).reserve_hint() == 4);

    ASSERT_SAME_TYPE(decltype(rev.reserve_hint()), std::size_t);
    ASSERT_SAME_TYPE(decltype(std::move(rev).reserve_hint()), std::size_t);
  }
  // Non-common, const bidirectional range.
  {
    const auto rev = std::ranges::reverse_view(BidirApproxSizedRange<Copyable>{buffer, 4});
    assert(std::ranges::reserve_hint(rev) == 4);
    assert(rev.reserve_hint() == 4);
    assert(std::move(rev).reserve_hint() == 4);

    ASSERT_SAME_TYPE(decltype(rev.reserve_hint()), std::size_t);
    ASSERT_SAME_TYPE(decltype(std::move(rev).reserve_hint()), std::size_t);
  }
  // Non-common, non-const (move only) bidirectional range.
  {
    auto rev = std::ranges::reverse_view(BidirApproxSizedRange<MoveOnly>{buffer, 4});
    assert(std::move(rev).reserve_hint() == 4);

    ASSERT_SAME_TYPE(decltype(std::move(rev).reserve_hint()), std::size_t);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
