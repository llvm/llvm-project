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
// friend constexpr iter_rvalue_reference_t<Iterator>
//   iter_move(const move_iterator& i)
//     noexcept(noexcept(ranges::iter_move(i.current))); // Since C++20

#include <iterator>

#include <cassert>
#include <type_traits>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"

int global;

template <bool IsNoexcept>
struct MaybeNoexceptMove {
  int x;
  using value_type = int;
  using difference_type = std::ptrdiff_t;

  constexpr friend value_type&& iter_move(MaybeNoexceptMove) noexcept(IsNoexcept) {
    return std::move(global);
  }

  int& operator*() const { static int a; return a; }

  MaybeNoexceptMove& operator++();
  MaybeNoexceptMove operator++(int);
};
using ThrowingBase = MaybeNoexceptMove<false>;
using NoexceptBase = MaybeNoexceptMove<true>;
static_assert(std::input_iterator<ThrowingBase>);
ASSERT_NOT_NOEXCEPT(std::ranges::iter_move(std::declval<ThrowingBase>()));
ASSERT_NOEXCEPT(std::ranges::iter_move(std::declval<NoexceptBase>()));

constexpr bool test() {
  // Can use `iter_move` with a regular array.
  {
    int a[] = {0, 1, 2};

    std::move_iterator<int*> i(a);
    static_assert(std::same_as<decltype(iter_move(i)), int&&>);
    assert(iter_move(i) == 0);

    ++i;
    assert(iter_move(i) == 1);
  }

  // Check that the `iter_move` customization point is being used.
  {
    int a[] = {0, 1, 2};

    int iter_move_invocations = 0;
    adl::Iterator base = adl::Iterator::TrackMoves(a, iter_move_invocations);
    std::move_iterator<adl::Iterator> i(base);
    int x = iter_move(i);
    assert(x == 0);
    assert(iter_move_invocations == 1);
  }

  // Check the `noexcept` specification.
  {
    using ThrowingIter = std::move_iterator<ThrowingBase>;
    ASSERT_NOT_NOEXCEPT(iter_move(std::declval<ThrowingIter>()));
    using NoexceptIter = std::move_iterator<NoexceptBase>;
    ASSERT_NOEXCEPT(iter_move(std::declval<NoexceptIter>()));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
