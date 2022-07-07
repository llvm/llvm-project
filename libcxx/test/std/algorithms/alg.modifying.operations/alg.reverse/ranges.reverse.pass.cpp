//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// <algorithm>

// template<bidirectional_iterator I, sentinel_for<I> S>
//   requires permutable<I>
//   constexpr I ranges::reverse(I first, S last);
// template<bidirectional_range R>
//   requires permutable<iterator_t<R>>
//   constexpr borrowed_iterator_t<R> ranges::reverse(R&& r);

#include <algorithm>
#include <array>
#include <concepts>
#include <ranges>

#include "almost_satisfies_types.h"
#include "MoveOnly.h"
#include "test_iterators.h"

template <class Iter, class Sent = sentinel_wrapper<Iter>>
concept HasReverseIt = requires (Iter first, Sent last) { std::ranges::reverse(first, last); };

static_assert(HasReverseIt<int*>);
static_assert(!HasReverseIt<BidirectionalIteratorNotDerivedFrom>);
static_assert(!HasReverseIt<BidirectionalIteratorNotDecrementable>);
static_assert(!HasReverseIt<PermutableNotForwardIterator>);
static_assert(!HasReverseIt<PermutableNotSwappable>);


template <class Range>
concept HasReverseR = requires (Range range) { std::ranges::reverse(range); };

static_assert(HasReverseR<UncheckedRange<int*>>);
static_assert(!HasReverseR<BidirectionalRangeNotDerivedFrom>);
static_assert(!HasReverseR<BidirectionalRangeNotDecrementable>);
static_assert(!HasReverseR<PermutableRangeNotForwardIterator>);
static_assert(!HasReverseR<PermutableRangeNotSwappable>);

template <class Iter, class Sent, size_t N>
constexpr void test(std::array<int, N> value, std::array<int, N> expected) {
  {
    auto val = value;
    std::same_as<Iter> decltype(auto) ret = std::ranges::reverse(Iter(val.data()), Sent(Iter(val.data() + val.size())));
    assert(val == expected);
    assert(base(ret) == val.data() + val.size());
  }
  {
    auto val = value;
    auto range = std::ranges::subrange(Iter(val.data()), Sent(Iter(val.data() + val.size())));
    std::same_as<Iter> decltype(auto) ret = std::ranges::reverse(range);
    assert(val == expected);
    assert(base(ret) == val.data() + val.size());
  }
}

template <class Iter, class Sent = Iter>
constexpr void test_iterators() {
  // simple test
  test<Iter, Sent, 4>({1, 2, 3, 4}, {4, 3, 2, 1});
  // check that an odd number of elements works
  test<Iter, Sent, 7>({1, 2, 3, 4, 5, 6, 7}, {7, 6, 5, 4, 3, 2, 1});
  // check that an empty range works
  test<Iter, Sent, 0>({}, {});
  // check that a single element works
  test<Iter, Sent, 1>({5}, {5});
}

struct SwapCounter {
  int* counter;
  constexpr SwapCounter(int* counter_) : counter(counter_) {}
  friend constexpr void swap(SwapCounter& lhs, SwapCounter&) { ++*lhs.counter; }
};

constexpr bool test() {
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>, sentinel_wrapper<bidirectional_iterator<int*>>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<random_access_iterator<int*>, sentinel_wrapper<random_access_iterator<int*>>>();
  test_iterators<contiguous_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>, sentinel_wrapper<contiguous_iterator<int*>>>();
  test_iterators<int*>();

  test_iterators<ProxyIterator<bidirectional_iterator<int*>>>();
  test_iterators<ProxyIterator<random_access_iterator<int*>>>();
  test_iterators<ProxyIterator<contiguous_iterator<int*>>>();

  // check that std::ranges::dangling is returned
  {
    [[maybe_unused]] std::same_as<std::ranges::dangling> auto ret = std::ranges::reverse(std::array {1, 2, 3, 4});
  }

  {
    {
      int counter = 0;
      SwapCounter a[] = {&counter, &counter, &counter, &counter};
      std::ranges::reverse(a);
      assert(counter == 2);
    }
    {
      int counter = 0;
      SwapCounter a[] = {&counter, &counter, &counter, &counter};
      std::ranges::reverse(a, a + 4);
      assert(counter == 2);
    }
  }

  // Move only types work for ProxyIterator
  {
    {
      MoveOnly a[] = {1, 2, 3};
      ProxyRange proxyA{a};
      std::ranges::reverse(proxyA.begin(), proxyA.end());
      assert(a[0].get() == 3);
      assert(a[1].get() == 2);
      assert(a[2].get() == 1);
    }
    {
      MoveOnly a[] = {1, 2, 3};
      ProxyRange proxyA{a};
      std::ranges::reverse(proxyA);
      assert(a[0].get() == 3);
      assert(a[1].get() == 2);
      assert(a[2].get() == 1);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
