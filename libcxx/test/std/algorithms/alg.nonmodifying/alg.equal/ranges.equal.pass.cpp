//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2,
//          class Pred = ranges::equal_to, class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_comparable<I1, I2, Pred, Proj1, Proj2>
//   constexpr bool ranges::equal(I1 first1, S1 last1, I2 first2, S2 last2,
//                                Pred pred = {},
//                                Proj1 proj1 = {}, Proj2 proj2 = {});
// template<input_range R1, input_range R2, class Pred = ranges::equal_to,
//          class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_comparable<iterator_t<R1>, iterator_t<R2>, Pred, Proj1, Proj2>
//   constexpr bool ranges::equal(R1&& r1, R2&& r2, Pred pred = {},
//                                Proj1 proj1 = {}, Proj2 proj2 = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class Iter1, class Sent1 = sentinel_wrapper<Iter1>,
          class Iter2 = Iter1, class Sent2 = sentinel_wrapper<Iter2>>
concept HasEqualIt = requires (Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2) {
  std::ranges::equal(first1, last1, first2, last2);
};

static_assert(HasEqualIt<int*>);
static_assert(!HasEqualIt<InputIteratorNotDerivedFrom>);
static_assert(!HasEqualIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasEqualIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasEqualIt<int*, int*, InputIteratorNotDerivedFrom>);
static_assert(!HasEqualIt<int*, int*, InputIteratorNotIndirectlyReadable>);
static_assert(!HasEqualIt<int*, int*, InputIteratorNotInputOrOutputIterator>);
static_assert(!HasEqualIt<int*, SentinelForNotSemiregular>);
static_assert(!HasEqualIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasEqualIt<int*, int*, int*, SentinelForNotSemiregular>);
static_assert(!HasEqualIt<int*, int*, int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasEqualIt<int*, int*, int**>);

template <class Range1, class Range2>
concept HasEqualR = requires (Range1 range1, Range2 range2) {
  std::ranges::equal(range1, range2);
};

static_assert(HasEqualR<UncheckedRange<int*>, UncheckedRange<int*>>);
static_assert(!HasEqualR<InputRangeNotDerivedFrom, UncheckedRange<int*>>);
static_assert(!HasEqualR<InputRangeNotIndirectlyReadable, UncheckedRange<int*>>);
static_assert(!HasEqualR<InputRangeNotInputOrOutputIterator, UncheckedRange<int*>>);
static_assert(!HasEqualR<InputRangeNotSentinelSemiregular, UncheckedRange<int*>>);
static_assert(!HasEqualR<InputRangeNotSentinelEqualityComparableWith, UncheckedRange<int*>>);
static_assert(!HasEqualR<UncheckedRange<int*>, InputRangeNotDerivedFrom>);
static_assert(!HasEqualR<UncheckedRange<int*>, InputRangeNotIndirectlyReadable>);
static_assert(!HasEqualR<UncheckedRange<int*>, InputRangeNotInputOrOutputIterator>);
static_assert(!HasEqualR<UncheckedRange<int*>, InputRangeNotSentinelSemiregular>);
static_assert(!HasEqualR<UncheckedRange<int*>, InputRangeNotSentinelEqualityComparableWith>);
static_assert(!HasEqualR<UncheckedRange<int*>, UncheckedRange<int**>>);

template <class Iter1, class Sent1, class Iter2, class Sent2 = Iter2>
constexpr void test_iterators() {
  { // simple test
    {
      int a[] = {1, 2, 3, 4};
      int b[] = {1, 2, 3, 4};
      std::same_as<bool> decltype(auto) ret = std::ranges::equal(Iter1(a), Sent1(Iter1(a + 4)),
                                                                 Iter2(b), Sent2(Iter2(b + 4)));
      assert(ret);
    }
    {
      int a[] = {1, 2, 3, 4};
      int b[] = {1, 2, 3, 4};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 4)));
      auto range2 = std::ranges::subrange(Iter2(b), Sent2(Iter2(b + 4)));
      std::same_as<bool> decltype(auto) ret = std::ranges::equal(range1, range2);
      assert(ret);
    }
  }

  { // check that false is returned for non-equal ranges
    {
      int a[] = {1, 2, 3, 4};
      int b[]  = {1, 2, 4, 4};
      assert(!std::ranges::equal(Iter1(a), Sent1(Iter1(a + 4)), Iter2(b), Sent2(Iter2(b + 4))));
    }
    {
      int a[] = {1, 2, 3, 4};
      int b[] = {1, 2, 4, 4};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 4)));
      auto range2 = std::ranges::subrange(Iter2(b), Sent2(Iter2(b + 4)));
      assert(!std::ranges::equal(range1, range2));
    }
  }

  { // check that the predicate is used (return true)
    {
      int a[] = {1, 2, 3, 4};
      int b[] = {2, 3, 4, 5};
      auto ret = std::ranges::equal(Iter1(a), Sent1(Iter1(a + 4)), Iter2(b), Sent2(Iter2(b + 4)),
                                    [](int l, int r) { return l != r; });
      assert(ret);
    }
    {
      int a[] = {1, 2, 3, 4};
      int b[] = {2, 3, 4, 5};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 4)));
      auto range2 = std::ranges::subrange(Iter2(b), Sent2(Iter2(b + 4)));
      auto ret = std::ranges::equal(range1, range2, [](int l, int r) { return l != r; });
      assert(ret);
    }
  }

  { // check that the predicate is used (return false)
    {
      int a[] = {1, 2, 3, 4};
      int b[] = {2, 3, 3, 5};
      auto ret = std::ranges::equal(Iter1(a), Sent1(Iter1(a + 4)), Iter2(b), Sent2(Iter2(b + 4)),
                                    [](int l, int r) { return l != r; });
      assert(!ret);
    }
    {
      int a[] = {1, 2, 3, 4};
      int b[] = {2, 3, 3, 5};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 4)));
      auto range2 = std::ranges::subrange(Iter2(b), Sent2(Iter2(b + 4)));
      auto ret = std::ranges::equal(range1, range2, [](int l, int r) { return l != r; });
      assert(!ret);
    }
  }

  { // check that the projections are used
    {
      int a[] = {1, 2, 3, 4, 5};
      int b[] = {6, 10, 14, 18, 22};
      auto ret = std::ranges::equal(Iter1(a), Sent1(Iter1(a + 5)),
                                    Iter2(b), Sent2(Iter2(b + 5)),
                                    {},
                                    [](int i) { return i * 4; },
                                    [](int i) { return i - 2; });
      assert(ret);
    }
    {
      int a[] = {1, 2, 3, 4, 5};
      int b[] = {6, 10, 14, 18, 22};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 5)));
      auto range2 = std::ranges::subrange(Iter2(b), Sent2(Iter2(b + 5)));
      auto ret = std::ranges::equal(range1,
                                    range2,
                                    {},
                                    [](int i) { return i * 4; },
                                    [](int i) { return i - 2; });
      assert(ret);
    }
  }

  { // check that different sized ranges work
    {
      int a[] = {4, 3, 2, 1};
      int b[] = {4, 3, 2, 1, 5, 6, 7};
      auto ret = std::ranges::equal(Iter1(a), Sent1(Iter1(a + 4)), Iter2(b), Sent2(Iter2(b + 7)));
      assert(!ret);
    }
    {
      int a[] = {4, 3, 2, 1};
      int b[] = {4, 3, 2, 1, 5, 6, 7};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 4)));
      auto range2 = std::ranges::subrange(Iter2(b), Sent2(Iter2(b + 7)));
      auto ret = std::ranges::equal(range1, range2);
      assert(!ret);
    }
  }

  { // check that two ranges with the same size but different values are different
    {
      int a[] = {4, 6, 34, 76, 5};
      int b[] = {4, 6, 34, 67, 5};
      auto ret = std::ranges::equal(Iter1(a), Sent1(Iter1(a + 5)), Iter2(b), Sent2(Iter2(b + 5)));
      assert(!ret);
    }
    {
      int a[] = {4, 6, 34, 76, 5};
      int b[] = {4, 6, 34, 67, 5};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 5)));
      auto range2 = std::ranges::subrange(Iter2(b), Sent2(Iter2(b + 5)));
      auto ret = std::ranges::equal(range1, range2);
      assert(!ret);
    }
  }

  { // check that two empty ranges work
    {
      std::array<int, 0> a = {};
      std::array<int, 0> b = {};
      auto ret = std::ranges::equal(Iter1(a.data()), Sent1(Iter1(a.data())), Iter2(b.data()), Sent2(Iter2(b.data())));
      assert(ret);
    }
    {
      std::array<int, 0> a = {};
      std::array<int, 0> b = {};
      auto range1          = std::ranges::subrange(Iter1(a.data()), Sent1(Iter1(a.data())));
      auto range2          = std::ranges::subrange(Iter2(b.data()), Sent2(Iter2(b.data())));
      auto ret = std::ranges::equal(range1, range2);
      assert(ret);
    }
  }

  { // check that it works with the first range empty
    {
      std::array<int, 0> a = {};
      int b[] = {1, 2};
      auto ret             = std::ranges::equal(Iter1(a.data()), Sent1(Iter1(a.data())), Iter2(b), Sent2(Iter2(b + 2)));
      assert(!ret);
    }
    {
      std::array<int, 0> a = {};
      int b[] = {1, 2};
      auto range1          = std::ranges::subrange(Iter1(a.data()), Sent1(Iter1(a.data())));
      auto range2 = std::ranges::subrange(Iter2(b), Sent2(Iter2(b + 2)));
      auto ret = std::ranges::equal(range1, range2);
      assert(!ret);
    }
  }

  { // check that it works with the second range empty
    {
      int a[] = {1, 2};
      std::array<int, 0> b = {};
      auto ret             = std::ranges::equal(Iter1(a), Sent1(Iter1(a + 2)), Iter2(b.data()), Sent2(Iter2(b.data())));
      assert(!ret);
    }
    {
      int a[] = {1, 2};
      std::array<int, 0> b = {};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 2)));
      auto range2          = std::ranges::subrange(Iter2(b.data()), Sent2(Iter2(b.data())));
      auto ret = std::ranges::equal(range1, range2);
      assert(!ret);
    }
  }
}

template<class Iter1, class Sent1 = Iter1>
constexpr void test_iterators2() {
  test_iterators<Iter1, Sent1, cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_iterators<Iter1, Sent1, cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators<Iter1, Sent1, forward_iterator<int*>>();
  test_iterators<Iter1, Sent1, bidirectional_iterator<int*>>();
  test_iterators<Iter1, Sent1, random_access_iterator<int*>>();
  test_iterators<Iter1, Sent1, contiguous_iterator<int*>>();
  test_iterators<Iter1, Sent1, int*>();
  test_iterators<Iter1, Sent1, const int*>();
}

constexpr bool test() {
  test_iterators2<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_iterators2<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators2<forward_iterator<int*>>();
  test_iterators2<bidirectional_iterator<int*>>();
  test_iterators2<random_access_iterator<int*>>();
  test_iterators2<contiguous_iterator<int*>>();
  test_iterators2<int*>();
  test_iterators2<const int*>();

  { // check that std::invoke is used
    struct S {
      constexpr S(int i_) : i(i_) {}
      constexpr bool equal(int o) { return i == o; }
      constexpr S& identity() { return *this; }
      int i;
    };
    {
      S a[] = {7, 8, 9};
      S b[] = {7, 8, 9};
      auto ret = std::ranges::equal(a, a + 3, b, b + 3, &S::equal, &S::identity, &S::i);
      assert(ret);
    }
    {
      S a[] = {7, 8, 9};
      S b[] = {7, 8, 9};
      auto ret = std::ranges::equal(a, b, &S::equal, &S::identity, &S::i);
      assert(ret);
    }
  }

  { // check that the complexity requirements are met
    { // different size
      {
        int a[] = {1, 2, 3};
        int b[] = {1, 2, 3, 4};
        int predCount = 0;
        int projCount = 0;
        auto pred = [&](int l, int r) { ++predCount; return l == r; };
        auto proj = [&](int i) { ++projCount; return i; };
        auto ret = std::ranges::equal(a, a + 3, b, b + 4, pred, proj, proj);
        assert(!ret);
        assert(predCount == 0);
        assert(projCount == 0);
      }
      {
        int a[] = {1, 2, 3};
        int b[] = {1, 2, 3, 4};
        int predCount = 0;
        int projCount = 0;
        auto pred = [&](int l, int r) { ++predCount; return l == r; };
        auto proj = [&](int i) { ++projCount; return i; };
        auto ret = std::ranges::equal(a, b, pred, proj, proj);
        assert(!ret);
        assert(predCount == 0);
        assert(projCount == 0);
      }
    }

    { // not a sized sentinel
      {
        int a[] = {1, 2, 3};
        int b[] = {1, 2, 3, 4};
        int predCount = 0;
        int projCount = 0;
        auto pred = [&](int l, int r) { ++predCount; return l == r; };
        auto proj = [&](int i) { ++projCount; return i; };
        auto ret = std::ranges::equal(a, sentinel_wrapper(a + 3), b, sentinel_wrapper(b + 4), pred, proj, proj);
        assert(!ret);
        assert(predCount <= 4);
        assert(projCount <= 7);
      }
      {
        int a[] = {1, 2, 3};
        int b[] = {1, 2, 3, 4};
        int predCount = 0;
        int projCount = 0;
        auto pred = [&](int l, int r) { ++predCount; return l == r; };
        auto proj = [&](int i) { ++projCount; return i; };
        auto range1 = std::ranges::subrange(a, sentinel_wrapper(a + 3));
        auto range2 = std::ranges::subrange(b, sentinel_wrapper(b + 4));
        auto ret = std::ranges::equal(range1, range2, pred, proj, proj);
        assert(!ret);
        assert(predCount <= 4);
        assert(projCount <= 7);
      }
    }

    { // same size
      {
        int a[] = {1, 2, 3};
        int b[] = {1, 2, 3};
        int predCount = 0;
        int projCount = 0;
        auto pred = [&](int l, int r) { ++predCount; return l == r; };
        auto proj = [&](int i) { ++projCount; return i; };
        auto ret = std::ranges::equal(a, a + 3, b, b + 3, pred, proj, proj);
        assert(ret);
        assert(predCount == 3);
        assert(projCount == 6);
      }
      {
        int a[] = {1, 2, 3};
        int b[] = {1, 2, 3};
        int predCount = 0;
        int projCount = 0;
        auto pred = [&](int l, int r) { ++predCount; return l == r; };
        auto proj = [&](int i) { ++projCount; return i; };
        auto ret = std::ranges::equal(a, b, pred, proj, proj);
        assert(ret);
        assert(predCount == 3);
        assert(projCount == 6);
      }
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
