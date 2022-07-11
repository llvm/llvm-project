//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<forward_iterator I, sentinel_for<I> S, weakly_incrementable O>
//   requires indirectly_copyable<I, O>
//   constexpr ranges::rotate_copy_result<I, O>
//     ranges::rotate_copy(I first, I middle, S last, O result);
// template<forward_range R, weakly_incrementable O>
//   requires indirectly_copyable<iterator_t<R>, O>
//   constexpr ranges::rotate_copy_result<borrowed_iterator_t<R>, O>
//     ranges::rotate_copy(R&& r, iterator_t<R> middle, O result);

// <algorithm>

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class Iter, class Out = int*, class Sent = sentinel_wrapper<Iter>>
concept HasRotateCopyIt = requires(Iter first, Sent last, Out out) { std::ranges::rotate_copy(first, first, last, out); };

template <class Range, class Out = int*>
concept HasRotateCopyR = requires(Range range, Out out) { std::ranges::rotate_copy(range, nullptr, out); };

static_assert(HasRotateCopyIt<int*>);
static_assert(!HasRotateCopyIt<BidirectionalIteratorNotDerivedFrom>);
static_assert(!HasRotateCopyIt<BidirectionalIteratorNotDecrementable>);
static_assert(!HasRotateCopyIt<int*, SentinelForNotSemiregular>);
static_assert(!HasRotateCopyIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasRotateCopyIt<int*, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasRotateCopyIt<int*, OutputIteratorNotInputOrOutputIterator>);

static_assert(HasRotateCopyR<UncheckedRange<int*>>);
static_assert(!HasRotateCopyR<BidirectionalRangeNotDerivedFrom>);
static_assert(!HasRotateCopyR<BidirectionalRangeNotDecrementable>);
static_assert(!HasRotateCopyR<UncheckedRange<int*, SentinelForNotSemiregular>>);
static_assert(!HasRotateCopyR<UncheckedRange<int*>, OutputIteratorNotIndirectlyWritable>);
static_assert(!HasRotateCopyR<UncheckedRange<int*>, OutputIteratorNotInputOrOutputIterator>);

static_assert(std::is_same_v<std::ranges::rotate_copy_result<int, int>, std::ranges::in_out_result<int, int>>);

template <class Iter, class OutIter, class Sent, int N>
constexpr void test(std::array<int, N> value, size_t middle, std::array<int, N> expected) {
  {
    std::array<int, N> out;
    std::same_as<std::ranges::in_out_result<Iter, OutIter>> decltype(auto) ret =
        std::ranges::rotate_copy(Iter(value.data()),
                                 Iter(value.data() + middle),
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
        std::ranges::rotate_copy(range, Iter(value.data() + middle), OutIter(out.data()));
    assert(base(ret.in) == value.data() + value.size());
    assert(base(ret.out) == out.data() + out.size());
    assert(out == expected);
  }
}

template <class Iter, class OutIter, class Sent>
constexpr void test_iterators() {
  // simple test
  test<Iter, OutIter, Sent, 4>({1, 2, 3, 4}, 2, {3, 4, 1, 2});

  // check that an empty range works
  test<Iter, OutIter, Sent, 0>({}, 0, {});

  // check that a single element range works
  test<Iter, OutIter, Sent, 1>({1}, 0, {1});

  // check that a two element range works
  test<Iter, OutIter, Sent, 2>({1, 2}, 1, {2, 1});

  // rotate on the first element
  test<Iter, OutIter, Sent, 7>({1, 2, 3, 4, 5, 6, 7}, 0, {1, 2, 3, 4, 5, 6, 7});

  // rotate on the second element
  test<Iter, OutIter, Sent, 7>({1, 2, 3, 4, 5, 6, 7}, 1, {2, 3, 4, 5, 6, 7, 1});

  // rotate on the last element
  test<Iter, OutIter, Sent, 7>({1, 2, 3, 4, 5, 6, 7}, 6, {7, 1, 2, 3, 4, 5, 6});

  // rotate on the one-past-the-end pointer
  test<Iter, OutIter, Sent, 7>({1, 2, 3, 4, 5, 6, 7}, 7, {1, 2, 3, 4, 5, 6, 7});
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
      std::ranges::rotate_copy(a, a + 2, a + 4, b);
      assert(c == 4);
    }
    {
      int c = 0;
      AssignmentCounter a[] = {&c, &c, &c, &c};
      AssignmentCounter b[] = {&c, &c, &c, &c};
      std::ranges::rotate_copy(a, a + 2, b);
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
