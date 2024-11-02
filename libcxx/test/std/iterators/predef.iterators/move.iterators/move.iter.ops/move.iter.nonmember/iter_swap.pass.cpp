//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <iterator>
//
// template<indirectly_swappable<Iterator> Iterator2>
//   friend constexpr void
//     iter_swap(const move_iterator& x, const move_iterator<Iterator2>& y)
//       noexcept(noexcept(ranges::iter_swap(x.current, y.current))); // Since C++20

#include <iterator>

#include <cassert>
#include <type_traits>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"

template <bool IsNoexcept>
struct MaybeNoexceptSwap {
  using value_type = int;
  using difference_type = ptrdiff_t;

  constexpr friend void iter_swap(MaybeNoexceptSwap, MaybeNoexceptSwap) noexcept(IsNoexcept) {
  }

  int& operator*() const { static int x; return x; }

  MaybeNoexceptSwap& operator++();
  MaybeNoexceptSwap operator++(int);
};
using ThrowingBase = MaybeNoexceptSwap<false>;
using NoexceptBase = MaybeNoexceptSwap<true>;
static_assert(std::input_iterator<ThrowingBase>);
ASSERT_NOT_NOEXCEPT(std::ranges::iter_swap(std::declval<ThrowingBase>(), std::declval<ThrowingBase>()));
ASSERT_NOEXCEPT(std::ranges::iter_swap(std::declval<NoexceptBase>(), std::declval<NoexceptBase>()));

constexpr bool test() {
  // Can use `iter_swap` with a regular array.
  {
    int a[] = {0, 1, 2};

    std::move_iterator b(a);
    std::move_iterator e(a + 2);
    assert(a[0] == 0);
    assert(a[2] == 2);

    static_assert(std::same_as<decltype(iter_swap(b, e)), void>);
    iter_swap(b, e);
    assert(a[0] == 2);
    assert(a[2] == 0);
  }

  // Check that the `iter_swap` customization point is being used.
  {
    int iter_swap_invocations = 0;
    adl::Iterator base1 = adl::Iterator::TrackSwaps(iter_swap_invocations);
    adl::Iterator base2 = adl::Iterator::TrackSwaps(iter_swap_invocations);
    std::move_iterator<adl::Iterator> i1(base1), i2(base2);
    iter_swap(i1, i2);
    assert(iter_swap_invocations == 1);

    iter_swap(i2, i1);
    assert(iter_swap_invocations == 2);
  }

  // Check the `noexcept` specification.
  {
    using ThrowingIter = std::move_iterator<ThrowingBase>;
    ASSERT_NOT_NOEXCEPT(iter_swap(std::declval<ThrowingIter>(), std::declval<ThrowingIter>()));
    using NoexceptIter = std::move_iterator<NoexceptBase>;
    ASSERT_NOEXCEPT(iter_swap(std::declval<NoexceptIter>(), std::declval<NoexceptIter>()));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
