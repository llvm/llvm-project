//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr take_view(V base, range_difference_t<V> count); // explicit since C++23

#include <cassert>
#include <ranges>

#include "test_convertible.h"
#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"
#include "types.h"

// SFINAE tests.

#if TEST_STD_VER >= 23

static_assert(!test_convertible<std::ranges::take_view<View>, View, std::ranges::range_difference_t<View>>(),
              "This constructor must be explicit");

#else

static_assert(test_convertible<std::ranges::take_view<View>, View, std::ranges::range_difference_t<View>>(),
              "This constructor must be explicit");

#endif // TEST_STD_VER >= 23

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    std::ranges::take_view<CopyableView> tv(CopyableView{buffer}, 0);
    assert(tv.base().ptr_ == buffer);
    assert(tv.begin() == tv.end()); // Checking we have correct size.
  }

  {
    std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 1);
    assert(std::move(tv).base().ptr_ == buffer);
    assert(std::ranges::next(tv.begin(), 1) == tv.end()); // Checking we have correct size.
  }

  {
    const std::ranges::take_view<CopyableView> tv(CopyableView{buffer}, 2);
    assert(tv.base().ptr_ == buffer);
    assert(std::ranges::next(tv.begin(), 2) == tv.end()); // Checking we have correct size.
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
