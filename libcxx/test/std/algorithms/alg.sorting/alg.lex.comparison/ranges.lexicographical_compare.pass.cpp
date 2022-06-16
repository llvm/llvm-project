//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2,
//          class Proj1 = identity, class Proj2 = identity,
//          indirect_strict_weak_order<projected<I1, Proj1>,
//                                     projected<I2, Proj2>> Comp = ranges::less>
//   constexpr bool
//     ranges::lexicographical_compare(I1 first1, S1 last1, I2 first2, S2 last2,
//                                     Comp comp = {}, Proj1 proj1 = {}, Proj2 proj2 = {});
// template<input_range R1, input_range R2, class Proj1 = identity,
//          class Proj2 = identity,
//          indirect_strict_weak_order<projected<iterator_t<R1>, Proj1>,
//                                     projected<iterator_t<R2>, Proj2>> Comp = ranges::less>
//   constexpr bool
//     ranges::lexicographical_compare(R1&& r1, R2&& r2, Comp comp = {},
//                                     Proj1 proj1 = {}, Proj2 proj2 = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

template <class Iter1, class Sent1 = Iter1, class Iter2 = int*, class Sent2 = int*>
concept HasLexicographicalCompareIt = requires(Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2) {
  std::ranges::lexicographical_compare(first1, last1, first2, last2);
};

template <class Range1, class Range2 = UncheckedRange<int*>>
concept HasLexicographicalCompareR = requires(Range1 range1, Range2 range2) {
  std::ranges::lexicographical_compare(range1, range2);
};

static_assert(HasLexicographicalCompareIt<int*>);
static_assert(!HasLexicographicalCompareIt<InputIteratorNotDerivedFrom>);
static_assert(!HasLexicographicalCompareIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasLexicographicalCompareIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasLexicographicalCompareIt<int*, SentinelForNotSemiregular>);
static_assert(!HasLexicographicalCompareIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasLexicographicalCompareIt<int*, int*, InputIteratorNotDerivedFrom>);
static_assert(!HasLexicographicalCompareIt<int*, int*, InputIteratorNotIndirectlyReadable>);
static_assert(!HasLexicographicalCompareIt<int*, int*, InputIteratorNotInputOrOutputIterator>);
static_assert(!HasLexicographicalCompareIt<int*, int*, int*, SentinelForNotSemiregular>);
static_assert(!HasLexicographicalCompareIt<int*, int*, int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasLexicographicalCompareIt<int*, int*, int**, int**>); // not indirect_strict_weak_order

static_assert(HasLexicographicalCompareR<UncheckedRange<int*>>);
static_assert(!HasLexicographicalCompareR<InputRangeNotDerivedFrom>);
static_assert(!HasLexicographicalCompareR<InputRangeNotIndirectlyReadable>);
static_assert(!HasLexicographicalCompareR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasLexicographicalCompareR<InputRangeNotSentinelSemiregular>);
static_assert(!HasLexicographicalCompareR<InputRangeNotSentinelEqualityComparableWith>);
static_assert(!HasLexicographicalCompareR<UncheckedRange<int*>, InputRangeNotDerivedFrom>);
static_assert(!HasLexicographicalCompareR<UncheckedRange<int*>, InputRangeNotIndirectlyReadable>);
static_assert(!HasLexicographicalCompareR<UncheckedRange<int*>, InputRangeNotInputOrOutputIterator>);
static_assert(!HasLexicographicalCompareR<UncheckedRange<int*>, InputRangeNotSentinelSemiregular>);
static_assert(!HasLexicographicalCompareR<UncheckedRange<int*>, InputRangeNotSentinelEqualityComparableWith>);
static_assert(!HasLexicographicalCompareIt<UncheckedRange<int*>, UncheckedRange<int**>>); // not indirect_strict_weak_order

template <int N, int M>
struct Data {
  std::array<int, N> input1;
  std::array<int, M> input2;
  bool expected;
};

template <class Iter1, class Sent1, class Iter2, class Sent2, int N, int M>
constexpr void test(Data<N, M> d) {
  {
    std::same_as<bool> decltype(auto) ret =
        std::ranges::lexicographical_compare(Iter1(d.input1.data()), Sent1(Iter1(d.input1.data() + d.input1.size())),
                                             Iter2(d.input2.data()), Sent2(Iter2(d.input2.data() + d.input2.size())));
    assert(ret == d.expected);
  }
  {
    auto range1 = std::ranges::subrange(Iter1(d.input1.data()), Sent1(Iter1(d.input1.data() + d.input1.size())));
    auto range2 = std::ranges::subrange(Iter2(d.input2.data()), Sent2(Iter2(d.input2.data() + d.input2.size())));
    std::same_as<bool> decltype(auto) ret =
        std::ranges::lexicographical_compare(range1, range2);
    assert(ret == d.expected);
  }
}

template <class Iter1, class Sent1, class Iter2, class Sent2 = Iter2>
constexpr void test_iterators() {
  // simple test
  test<Iter1, Sent1, Iter2, Sent2, 4, 4>({.input1 = {1, 2}, .input2 = {1, 2, 3, 4}, .expected = true});
  // ranges are identical
  test<Iter1, Sent1, Iter2, Sent2, 4, 4>({.input1 = {1, 2, 3, 4}, .input2 = {1, 2, 3, 4}, .expected = false});
  // first range is empty
  test<Iter1, Sent1, Iter2, Sent2, 0, 4>({.input1 = {}, .input2 = {1, 2, 3, 4}, .expected = true});
  // second range is empty
  test<Iter1, Sent1, Iter2, Sent2, 4, 0>({.input1 = {1, 2, 3, 4}, .input2 = {}, .expected = false});
  // both ranges are empty
  test<Iter1, Sent1, Iter2, Sent2, 0, 0>({.input1 = {}, .input2 = {}, .expected = false});
  // the first range compares less; first range is smaller
  test<Iter1, Sent1, Iter2, Sent2, 3, 5>({.input1 = {1, 2, 3}, .input2 = {1, 2, 4, 5, 6}, .expected = true});
  // the second range compares less; first range is smaller
  test<Iter1, Sent1, Iter2, Sent2, 3, 5>({.input1 = {1, 2, 4}, .input2 = {1, 2, 3, 4, 5}, .expected = false});
  // the first range compares less; second range is smaller
  test<Iter1, Sent1, Iter2, Sent2, 5, 3>({.input1 = {1, 2, 3, 4, 5}, .input2 = {1, 2, 4}, .expected = true});
  // the second range compares less; second range is smaller
  test<Iter1, Sent1, Iter2, Sent2, 5, 3>({.input1 = {1, 2, 4, 5, 6}, .input2 = {1, 2, 3}, .expected = false});
}

template <class Iter1, class Sent1 = Iter1>
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

  { // check that custom projections and the comparator are used properly
    {
      int a[] = {3, 4, 5, 6};
      int b[] = {24, 33, 42, 51};

      auto ret = std::ranges::lexicographical_compare(std::begin(a), std::end(a),
                                                      std::begin(b), std::end(b),
                                                      [](int lhs, int rhs) { return lhs == rhs + 5; },
                                                      [](int v) { return v - 2; },
                                                      [](int v) { return v / 3; });
      assert(!ret);
    }
    {
      int a[] = {3, 4, 5, 6};
      int b[] = {24, 33, 42, 51};

      auto ret = std::ranges::lexicographical_compare(a, b,
                                                      [](int lhs, int rhs) { return lhs == rhs + 5; },
                                                      [](int v) { return v - 2; },
                                                      [](int v) { return v / 3; });
      assert(!ret);
    }
  }

  { // check that std::invoke is used
    struct S {
      constexpr S(int i_) : i(i_) {}
      constexpr bool compare(const S& j) const { return j.i < i; }
      constexpr const S& identity() const { return *this; }
      int i;
    };
    {
      S a[] = {1, 2, 3, 4};
      auto ret = std::ranges::lexicographical_compare(std::begin(a), std::end(a),
                                                      std::begin(a), std::end(a),
                                                      &S::compare,
                                                      &S::identity,
                                                      &S::identity);
      assert(!ret);
    }
    {
      S a[] = {1, 2, 3, 4};
      auto ret = std::ranges::lexicographical_compare(a, a, &S::compare, &S::identity, &S::identity);
      assert(!ret);
    }
  }

  { // check that the implicit conversion to bool works
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::lexicographical_compare(std::begin(a), std::end(a),
                                                      std::begin(a), std::end(a),
                                                      [](int i, int j) { return BooleanTestable{i < j}; });
      assert(!ret);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::lexicographical_compare(a, a, [](int i, int j) { return BooleanTestable{i < j}; });
      assert(!ret);
    }
  }

  { // check that the complexity requirements are met
    {
      int predCount = 0;
      auto pred = [&](int i, int j) { ++predCount; return i < j; };
      auto proj1Count = 0;
      auto proj1 = [&](int i) { ++proj1Count; return i; };
      auto proj2Count = 0;
      auto proj2 = [&](int i) { ++proj2Count; return i; };
      int a[] = {1, 2, 3, 4, 5};
      auto ret = std::ranges::lexicographical_compare(std::begin(a), std::end(a), std::begin(a), std::end(a), pred, proj1, proj2);
      assert(!ret);
      assert(predCount == 10);
      assert(proj1Count == 10);
      assert(proj2Count == 10);
    }
    {
      int predCount = 0;
      auto pred = [&](int i, int j) { ++predCount; return i < j; };
      auto proj1Count = 0;
      auto proj1 = [&](int i) { ++proj1Count; return i; };
      auto proj2Count = 0;
      auto proj2 = [&](int i) { ++proj2Count; return i; };
      int a[] = {1, 2, 3, 4, 5};
      auto ret = std::ranges::lexicographical_compare(a, a, pred, proj1, proj2);
      assert(!ret);
      assert(predCount == 10);
      assert(proj1Count == 10);
      assert(proj2Count == 10);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
