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

// constexpr auto end() requires (!simple-view<V>)
// constexpr auto end() const requires range<const V>

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "test_range.h"
#include "types.h"

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  // sized_range && random_access_iterator
  {
    std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView{buffer}, 0);
    assert(tv.end() == std::ranges::next(tv.begin(), 0));
    ASSERT_SAME_TYPE(decltype(tv.end()), RandomAccessIter);
  }

  {
    const std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView{buffer}, 1);
    assert(tv.end() == std::ranges::next(tv.begin(), 1));
    ASSERT_SAME_TYPE(decltype(tv.end()), RandomAccessIter);
  }

  // sized_range && !random_access_iterator
  {
    std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 2);
    assert(tv.end() == std::ranges::next(tv.begin(), 2));
    ASSERT_SAME_TYPE(decltype(tv.end()), std::default_sentinel_t);
  }

  {
    const std::ranges::take_view<SizedForwardView> tv(SizedForwardView{buffer}, 3);
    assert(tv.end() == std::ranges::next(tv.begin(), 3));
    ASSERT_SAME_TYPE(decltype(tv.end()), std::default_sentinel_t);
  }

  // !sized_range
  {
    std::ranges::take_view<ContiguousView> tv(ContiguousView{buffer}, 4);
    assert(tv.end() == std::ranges::next(tv.begin(), 4));

    // The <sentinel> type.
    static_assert(!std::same_as<decltype(tv.end()), std::default_sentinel_t>);
    static_assert(!std::same_as<decltype(tv.end()), int*>);
  }

  {
    const std::ranges::take_view<ContiguousView> tv(ContiguousView{buffer}, 5);
    assert(tv.end() == std::ranges::next(tv.begin(), 5));
  }

  // Just to cover the case where count == 8.
  {
    std::ranges::take_view<SizedRandomAccessView> tv(SizedRandomAccessView{buffer}, 8);
    assert(tv.end() == std::ranges::next(tv.begin(), 8));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
