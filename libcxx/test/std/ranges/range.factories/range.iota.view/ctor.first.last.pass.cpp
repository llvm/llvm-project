//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iota_view(iterator first, see below last); // explicit since C++23

#include <ranges>
#include <cassert>

#include "test_convertible.h"
#include "test_macros.h"
#include "types.h"

// SFINAE tests.

#if TEST_STD_VER >= 23

std::ranges::iota_view<SomeInt, SomeInt> view;

static_assert(!test_convertible<std::ranges::iota_view<SomeInt, SomeInt>,
                                decltype(std::ranges::iota_view<SomeInt, SomeInt>{}.begin()),
                                decltype(std::ranges::iota_view<SomeInt, SomeInt>{}.end())>(),
              "This constructor must be explicit");

static_assert(!test_convertible<std::ranges::iota_view<SomeInt>,
                                decltype(std::ranges::iota_view{SomeInt{0}}.begin()),
                                decltype(std::unreachable_sentinel)>(),
              "This constructor must be explicit");

static_assert(!test_convertible<std::ranges::iota_view<SomeInt, IntComparableWith<SomeInt>>,
                                decltype(std::ranges::iota_view{SomeInt(0), IntComparableWith(SomeInt(10))}.begin()),
                                decltype(std::ranges::iota_view{SomeInt(0), IntComparableWith(SomeInt(10))}.end())>(),
              "This constructor must be explicit");

#else

static_assert(test_convertible<std::ranges::iota_view<SomeInt, SomeInt>,
                               decltype(std::ranges::iota_view<SomeInt, SomeInt>{}.begin()),
                               decltype(std::ranges::iota_view<SomeInt, SomeInt>{}.end())>(),
              "This constructor must not be explicit");

static_assert(test_convertible<std::ranges::iota_view<SomeInt>,
                               decltype(std::ranges::iota_view{SomeInt{0}}.begin()),
                               decltype(std::unreachable_sentinel)>(),
              "This constructor must not be explicit");

static_assert(test_convertible<std::ranges::iota_view<SomeInt, IntComparableWith<SomeInt>>,
                               decltype(std::ranges::iota_view{SomeInt(0), IntComparableWith(SomeInt(10))}.begin()),
                               decltype(std::ranges::iota_view{SomeInt(0), IntComparableWith(SomeInt(10))}.end())>(),
              "This constructor must not be explicit");

#endif // TEST_STD_VER >= 23

constexpr bool test() {
  {
    std::ranges::iota_view commonView(SomeInt(0), SomeInt(10));
    std::ranges::iota_view<SomeInt, SomeInt> io(commonView.begin(), commonView.end());
    assert(std::ranges::next(io.begin(), 10) == io.end());
  }

  {
    std::ranges::iota_view unreachableSent(SomeInt(0));
    std::ranges::iota_view<SomeInt> io(unreachableSent.begin(), std::unreachable_sentinel);
    assert(std::ranges::next(io.begin(), 10) != io.end());
  }

  {
    std::ranges::iota_view differentTypes(SomeInt(0), IntComparableWith(SomeInt(10)));
    std::ranges::iota_view<SomeInt, IntComparableWith<SomeInt>> io(differentTypes.begin(), differentTypes.end());
    assert(std::ranges::next(io.begin(), 10) == io.end());
  }

  {
    std::ranges::iota_view<int, std::unreachable_sentinel_t> iv1;
    // There should be only one overload available and {} resolves to unreachable_sentinel_t
    [[maybe_unused]] std::ranges::iota_view<int, std::unreachable_sentinel_t> iv2(iv1.begin(), {});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}

