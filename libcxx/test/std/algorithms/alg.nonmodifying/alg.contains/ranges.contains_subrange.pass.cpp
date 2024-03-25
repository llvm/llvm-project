//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=2000000

// template<forward_iterator I1, sentinel_for<I1> S1,
//          forward_iterator I2, sentinel_for<I2> S2, class Proj = identity>
//     requires indirectly_comparable<I1, I2, Pred, Proj1, Proj2>
//     constexpr bool ranges::contains_subrange(I1 first1, S1 last1, I2 first2, S2 last2,
//                                     Pred pred = {}, Proj1 proj1 = {}, Proj2 proj2 = {});       // since C++23

// template<forward_range R1, forward_range R2,
//          class Pred = ranges::equal_to, class Proj1 = identity, class Proj2 = identity>
//     requires indirectly_comparable<iterator_t<R1>, iterator_t<R2>, Pred, Proj1, Proj2>
//     constexpr bool ranges::contains_subrange(R1&& r1, R2&& r2, Pred pred = {},
//                                              Proj1 proj1 = {}, Proj2 proj2 = {});                 // since C++23

#include <algorithm>
#include <cassert>
#include <concepts>
#include <ranges>
#include <utility>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct NotEqualityComparable {};

template <class Iter1, class Sent1 = Iter1, class Iter2 = int*, class Sent2 = Iter2>
concept HasContainsSubrangeIt = requires(Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2) {
  std::ranges::contains_subrange(first1, last1, first2, last2);
};

static_assert(HasContainsSubrangeIt<int*>);
static_assert(!HasContainsSubrangeIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasContainsSubrangeIt<ForwardIteratorNotIncrementable>);
static_assert(!HasContainsSubrangeIt<int*, SentinelForNotSemiregular>);
static_assert(!HasContainsSubrangeIt<int*, int*, int**>); // not indirectly comparable
static_assert(!HasContainsSubrangeIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasContainsSubrangeIt<int*, int*, ForwardIteratorNotDerivedFrom>);
static_assert(!HasContainsSubrangeIt<int*, int*, ForwardIteratorNotIncrementable>);
static_assert(!HasContainsSubrangeIt<int*, int*, int*, SentinelForNotSemiregular>);
static_assert(!HasContainsSubrangeIt<int*, int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Range1, class Range2 = UncheckedRange<int*>>
concept HasContainsSubrangeR = requires(Range1&& range1, Range2&& range2) {
  std::ranges::contains_subrange(std::forward<Range1>(range1), std::forward<Range2>(range2));
};

static_assert(HasContainsSubrangeR<UncheckedRange<int*>>);
static_assert(!HasContainsSubrangeR<ForwardRangeNotDerivedFrom>);
static_assert(!HasContainsSubrangeR<ForwardIteratorNotIncrementable>);
static_assert(!HasContainsSubrangeR<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasContainsSubrangeR<ForwardRangeNotSentinelEqualityComparableWith>);
static_assert(!HasContainsSubrangeR<UncheckedRange<int*>, UncheckedRange<int**>>); // not indirectly comparable
static_assert(!HasContainsSubrangeR<UncheckedRange<int*>, ForwardRangeNotDerivedFrom>);
static_assert(!HasContainsSubrangeR<UncheckedRange<int*>, ForwardRangeNotIncrementable>);
static_assert(!HasContainsSubrangeR<UncheckedRange<int*>, ForwardRangeNotSentinelSemiregular>);
static_assert(!HasContainsSubrangeR<UncheckedRange<int*>, ForwardRangeNotSentinelEqualityComparableWith>);

template <class Iter1, class Sent1 = Iter1, class Iter2, class Sent2 = Iter2>
constexpr void test_iterators() {
  { // simple tests
    int a[]       = {1, 2, 3, 4, 5, 6};
    int p[]       = {3, 4, 5};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      std::same_as<bool> decltype(auto) ret =
          std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      std::same_as<bool> decltype(auto) ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // no match
    int a[]       = {1, 2, 3, 4, 5, 6};
    int p[]       = {3, 4, 2};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(!ret);
    }
  }

  { // range consists of just one element
    int a[]       = {3};
    int p[]       = {3, 4, 2};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(!ret);
    }
  }

  { // subrange consists of just one element
    int a[]       = {23, 1, 20, 3, 54, 2};
    int p[]       = {3};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // range has zero length
    int a[]       = {};
    int p[]       = {3, 4, 2};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(a)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(!ret);
    }
  }

  { // subrange has zero length
    int a[]       = {3, 4, 2};
    int p[]       = {};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // range and subrange both have zero length
    int a[]       = {};
    int p[]       = {};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(a)));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(p)));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // range and subrange are identical
    int a[]       = {3, 4, 11, 32, 54, 2};
    int p[]       = {3, 4, 11, 32, 54, 2};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // subrange is longer than range
    int a[]       = {3, 4, 2};
    int p[]       = {23, 3, 4, 2, 11, 32, 54, 2};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(!ret);
    }
  }

  { // subrange is the prefix
    int a[]       = {3, 43, 5, 100, 433, 278, 6457, 900};
    int p[]       = {3, 43, 5};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // subrange is the suffix
    int a[]       = {3, 43, 5, 7, 68, 100, 433, 900};
    int p[]       = {100, 433, 900};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // subrange is a subsequence
    int a[]       = {23, 1, 0, 54, 2};
    int p[]       = {1, 0, 2};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(!ret);
    }
  }

  { // repeated subrange
    int a[]       = {23, 1, 0, 2, 54, 1, 0, 2, 23, 33};
    int p[]       = {1, 0, 2};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange);
      assert(ret);
    }
  }

  { // check that the predicate is used
    int a[]       = {23, 81, 61, 0, 42, 25, 1, 2, 1, 29, 2};
    int p[]       = {-1, -2, -1};
    auto pred     = [](int l, int r) { return l * -1 == r; };
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    {
      bool ret = std::ranges::contains_subrange(whole.begin(), whole.end(), subrange.begin(), subrange.end(), pred);
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange, pred);
      assert(ret);
    }
  }

  { // check that the projections are used
    int a[]       = {1, 3, 15, 1, 2, 1, 8};
    int p[]       = {2, 1, 2};
    auto whole    = std::ranges::subrange(Iter1(a), Sent1(Iter1(std::end(a))));
    auto subrange = std::ranges::subrange(Iter2(p), Sent2(Iter2(std::end(p))));
    auto proj1    = [](int i) { return i - 3; };
    auto proj2    = [](int i) { return i * -1; };
    {
      bool ret = std::ranges::contains_subrange(
          whole.begin(), whole.end(), subrange.begin(), subrange.end(), {}, proj1, proj2);
      assert(ret);
    }
    {
      bool ret = std::ranges::contains_subrange(whole, subrange, {}, proj1, proj2);
      assert(ret);
    }
  }
}

constexpr bool test() {
  types::for_each(types::forward_iterator_list<int*>{}, []<class Iter1> {
    types::for_each(types::forward_iterator_list<int*>{}, []<class Iter2> {
      test_iterators<Iter1, Iter1, Iter2, Iter2>();
      test_iterators<Iter1, Iter1, Iter2, sized_sentinel<Iter2>>();
      test_iterators<Iter1, sized_sentinel<Iter1>, Iter2, Iter2>();
      test_iterators<Iter1, sized_sentinel<Iter1>, Iter2, sized_sentinel<Iter2>>();
    });
  });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
