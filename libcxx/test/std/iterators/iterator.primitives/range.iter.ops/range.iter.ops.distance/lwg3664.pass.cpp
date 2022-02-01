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

// template<class I, sentinel_for<I> S>
//   requires (!sized_sentinel_for<S, I>)
//     constexpr iter_difference_t<I> ranges::distance(I first, S last);
//
// template<class I, sized_sentinel_for<decay_t<I>> S>
//   constexpr iter_difference_t<I> ranges::distance(const I& first, S last);

#include <iterator>
#include <cassert>

#include "test_iterators.h"

template<class It>
struct EvilSentinel {
  It p_;
  friend constexpr bool operator==(EvilSentinel s, It p) { return s.p_ == p; }
  friend constexpr auto operator-(EvilSentinel s, It p) { return s.p_ - p; }
  friend constexpr auto operator-(It p, EvilSentinel s) { return p - s.p_; }
  friend constexpr void operator-(EvilSentinel s, int(&)[3]) = delete;
  friend constexpr void operator-(EvilSentinel s, int(&&)[3]) = delete;
  friend constexpr void operator-(EvilSentinel s, const int(&)[3]) = delete;
  friend constexpr void operator-(EvilSentinel s, const int(&&)[3]) = delete;
};
static_assert( std::sized_sentinel_for<EvilSentinel<int*>, int*>);
static_assert(!std::sized_sentinel_for<EvilSentinel<int*>, const int*>);
static_assert( std::sized_sentinel_for<EvilSentinel<const int*>, int*>);
static_assert( std::sized_sentinel_for<EvilSentinel<const int*>, const int*>);

constexpr bool test() {
  {
    int a[] = {1, 2, 3};
    assert(std::ranges::distance(a, a + 3) == 3);
    assert(std::ranges::distance(a, a) == 0);
    assert(std::ranges::distance(a + 3, a) == -3);
  }
  {
    int a[] = {1, 2, 3};
    assert(std::ranges::distance(a, EvilSentinel<int*>{a+3}) == 3);
    assert(std::ranges::distance(a, EvilSentinel<int*>{a}) == 0);
    assert(std::ranges::distance(a+3, EvilSentinel<int*>{a}) == -3);
    assert(std::ranges::distance(std::move(a), EvilSentinel<int*>{a+3}) == 3);
  }
  {
    const int a[] = {1, 2, 3};
    assert(std::ranges::distance(a, EvilSentinel<const int*>{a+3}) == 3);
    assert(std::ranges::distance(a, EvilSentinel<const int*>{a}) == 0);
    assert(std::ranges::distance(a+3, EvilSentinel<const int*>{a}) == -3);
    assert(std::ranges::distance(std::move(a), EvilSentinel<const int*>{a+3}) == 3);
    static_assert(!std::is_invocable_v<decltype(std::ranges::distance), const int(&)[3], EvilSentinel<int*>>);
    static_assert(!std::is_invocable_v<decltype(std::ranges::distance), const int(&&)[3], EvilSentinel<int*>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
