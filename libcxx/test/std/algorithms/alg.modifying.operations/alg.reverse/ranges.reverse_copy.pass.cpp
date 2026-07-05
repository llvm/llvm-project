//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<bidirectional_iterator I, sentinel_for<I> S, weakly_incrementable O>
//   requires indirectly_copyable<I, O>
//   constexpr ranges::reverse_copy_result<I, O>
//     ranges::reverse_copy(I first, S last, O result);
// template<bidirectional_range R, weakly_incrementable O>
//   requires indirectly_copyable<iterator_t<R>, O>
//   constexpr ranges::reverse_copy_result<borrowed_iterator_t<R>, O>
//     ranges::reverse_copy(R&& r, O result);

// <algorithm>

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class Iter, class Out = int*, class Sent = sentinel_wrapper<Iter>>
concept HasReverseCopyIt = requires(Iter first, Sent last, Out out) { std::ranges::reverse_copy(first, last, out); };

template <class Range, class Out = int*>
concept HasReverseCopyR = requires(Range range, Out out) { std::ranges::reverse_copy(range, out); };

static_assert(HasReverseCopyIt<int*>);
static_assert(!HasReverseCopyIt<BidirectionalIteratorNotDerivedFrom>);
static_assert(!HasReverseCopyIt<BidirectionalIteratorNotDecrementable>);
static_assert(!HasReverseCopyIt<int*, SentinelForNotSemiregular>);
static_assert(!HasReverseCopyIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasReverseCopyIt<int*, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasReverseCopyIt<int*, OutputIteratorNotInputOrOutputIterator>);

static_assert(HasReverseCopyR<UncheckedRange<int*>>);
static_assert(!HasReverseCopyR<BidirectionalRangeNotDerivedFrom>);
static_assert(!HasReverseCopyR<BidirectionalRangeNotDecrementable>);
static_assert(!HasReverseCopyR<UncheckedRange<int*, SentinelForNotSemiregular>>);
static_assert(!HasReverseCopyR<UncheckedRange<int*>, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasReverseCopyR<UncheckedRange<int*>, OutputIteratorNotInputOrOutputIterator>);

static_assert(std::is_same_v<std::ranges::reverse_copy_result<int, int>, std::ranges::in_out_result<int, int>>);

template <class Iter, class OutIter, class Sent, int N>
constexpr void test(std::array<int, N> value, std::array<int, N> expected) {
  {
    std::array<int, N> out;
    std::same_as<std::ranges::in_out_result<Iter, OutIter>> decltype(auto) ret =
        std::ranges::reverse_copy(Iter(value.data()),
                                 Sent(Iter(value.data() + value.size())),
                                 OutIter(out.data()));
    assert(base(ret.in) == value.data() + value.size());
    assert(base(ret.out) == out.data() + out.size());
    assert(out == expected);
  }
  {
    std::array<int, N> out;
    auto range = std::ranges::subrange(Iter(value.data()), Sent(Iter(value.data() + value.size())));
    std::same_as<std::ranges::in_out_result<Iter, OutIter>> decltype(auto) ret =
        std::ranges::reverse_copy(range, OutIter(out.data()));
    assert(base(ret.in) == value.data() + value.size());
    assert(base(ret.out) == out.data() + out.size());
    assert(out == expected);
  }
}

template <class Iter, class OutIter, class Sent>
constexpr void test_iterators() {
  // simple test
  test<Iter, OutIter, Sent, 4>({1, 2, 3, 4}, {4, 3, 2, 1});

  // check that an empty range works
  test<Iter, OutIter, Sent, 0>({}, {});

  // check that a single element range works
  test<Iter, OutIter, Sent, 1>({1}, {1});

  // check that a two element range works
  test<Iter, OutIter, Sent, 2>({1, 2}, {2, 1});
}

template <class Iter, class Sent = Iter>
constexpr void test_out_iterators() {
  test_iterators<Iter, cpp20_output_iterator<int*>, Sent>();
  test_iterators<Iter, forward_iterator<int*>, Sent>();
  test_iterators<Iter, bidirectional_iterator<int*>, Sent>();
  test_iterators<Iter, random_access_iterator<int*>, Sent>();
  test_iterators<Iter, contiguous_iterator<int*>, Sent>();
  test_iterators<Iter, int*, Sent>();
}

constexpr bool test() {
  test_out_iterators<bidirectional_iterator<int*>>();
  test_out_iterators<random_access_iterator<int*>>();
  test_out_iterators<contiguous_iterator<int*>>();
  test_out_iterators<int*>();
  test_out_iterators<const int*>();

  {
    struct AssignmentCounter {
      int* counter;

      constexpr AssignmentCounter(int* counter_) : counter(counter_) {}
      constexpr AssignmentCounter& operator=(const AssignmentCounter&) { ++*counter; return *this; }
    };

    {
      int c = 0;
      AssignmentCounter a[] = {&c, &c, &c, &c};
      AssignmentCounter b[] = {&c, &c, &c, &c};
      std::ranges::reverse_copy(a, a + 4, b);
      assert(c == 4);
    }
    {
      int c = 0;
      AssignmentCounter a[] = {&c, &c, &c, &c};
      AssignmentCounter b[] = {&c, &c, &c, &c};
      std::ranges::reverse_copy(a, b);
      assert(c == 4);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
