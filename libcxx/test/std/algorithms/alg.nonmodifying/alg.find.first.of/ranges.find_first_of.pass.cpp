//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_iterator I1, sentinel_for<I1> S1, forward_iterator I2, sentinel_for<I2> S2,
//          class Pred = ranges::equal_to, class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_comparable<I1, I2, Pred, Proj1, Proj2>
//   constexpr I1 ranges::find_first_of(I1 first1, S1 last1, I2 first2, S2 last2,
//                                      Pred pred = {},
//                                      Proj1 proj1 = {}, Proj2 proj2 = {});
// template<input_range R1, forward_range R2,
//          class Pred = ranges::equal_to, class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_comparable<iterator_t<R1>, iterator_t<R2>, Pred, Proj1, Proj2>
//   constexpr borrowed_iterator_t<R1>
//     ranges::find_first_of(R1&& r1, R2&& r2,
//                           Pred pred = {},
//                           Proj1 proj1 = {}, Proj2 proj2 = {});

#include <algorithm>
#include <array>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

template <class Iter1, class Iter2 = int*, class Sent1 = Iter1, class Sent2 = Iter2>
concept HasFindFirstOfIt = requires(Iter1 iter1, Sent1 sent1, Iter2 iter2, Sent2 sent2) {
                             std::ranges::find_first_of(iter1, sent1, iter2, sent2);
                           };

static_assert(HasFindFirstOfIt<int*>);
static_assert(!HasFindFirstOfIt<InputIteratorNotDerivedFrom>);
static_assert(!HasFindFirstOfIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasFindFirstOfIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasFindFirstOfIt<int*, ForwardIteratorNotDerivedFrom>);
static_assert(!HasFindFirstOfIt<int*, ForwardIteratorNotIncrementable>);
static_assert(!HasFindFirstOfIt<int*, int*, SentinelForNotSemiregular>);
static_assert(!HasFindFirstOfIt<int*, int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasFindFirstOfIt<int*, int*, int*, SentinelForNotSemiregular>);
static_assert(!HasFindFirstOfIt<int*, int*, int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasFindFirstOfIt<int*, int**>); // not indirectly_comparable

template <class Range1, class Range2 = UncheckedRange<int*>>
concept HasFindFirstOfR = requires(Range1 range1, Range2 range2) {
                             std::ranges::find_first_of(range1, range2);
                           };

static_assert(HasFindFirstOfR<UncheckedRange<int*>>);
static_assert(!HasFindFirstOfR<InputRangeNotDerivedFrom>);
static_assert(!HasFindFirstOfR<InputRangeNotIndirectlyReadable>);
static_assert(!HasFindFirstOfR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasFindFirstOfR<UncheckedRange<int*>, ForwardRangeNotDerivedFrom>);
static_assert(!HasFindFirstOfR<UncheckedRange<int*>, ForwardRangeNotIncrementable>);
static_assert(!HasFindFirstOfR<UncheckedRange<int*>, InputRangeNotSentinelSemiregular>);
static_assert(!HasFindFirstOfR<UncheckedRange<int*>, InputRangeNotSentinelEqualityComparableWith>);
static_assert(!HasFindFirstOfR<UncheckedRange<int*>, ForwardRangeNotSentinelSemiregular>);
static_assert(!HasFindFirstOfR<UncheckedRange<int*>, ForwardRangeNotSentinelEqualityComparableWith>);
static_assert(!HasFindFirstOfR<UncheckedRange<int*>, UncheckedRange<int**>>); // not indirectly_comparable

template <int N1, int N2>
struct Data {
  std::array<int, N1> input1;
  std::array<int, N2> input2;
  std::ptrdiff_t expected;
};

template <class Iter1, class Sent1, class Iter2, class Sent2, int N1, int N2>
constexpr void test(Data<N1, N2> d) {
  {
    std::same_as<Iter1> decltype(auto) ret =
        std::ranges::find_first_of(Iter1(d.input1.data()), Sent1(Iter1(d.input1.data() + d.input1.size())),
                                   Iter2(d.input2.data()), Sent2(Iter2(d.input2.data() + d.input2.size())));
    assert(base(ret) == d.input1.data() + d.expected);
  }
  {
    auto range1 = std::ranges::subrange(Iter1(d.input1.data()), Sent1(Iter1(d.input1.data() + d.input1.size())));
    auto range2 = std::ranges::subrange(Iter2(d.input2.data()), Sent2(Iter2(d.input2.data() + d.input2.size())));
    std::same_as<Iter1> decltype(auto) ret = std::ranges::find_first_of(range1, range2);
    assert(base(ret) == d.input1.data() + d.expected);
  }
}

template <class Iter1, class Sent1, class Iter2, class Sent2 = Iter2>
constexpr void test_iterators() {
  // simple test
  test<Iter1, Sent1, Iter2, Sent2, 4, 2>({.input1 = {1, 2, 3, 4}, .input2 = {2, 3}, .expected = 1});
  // other elements from input2 are checked
  test<Iter1, Sent1, Iter2, Sent2, 4, 2>({.input1 = {1, 2, 3, 4}, .input2 = {3, 2}, .expected = 1});
  // an empty second range returns last
  test<Iter1, Sent1, Iter2, Sent2, 4, 0>({.input1 = {1, 2, 3, 4}, .input2 = {}, .expected = 4});
  // check that an empty first range works
  test<Iter1, Sent1, Iter2, Sent2, 0, 1>({.input1 = {}, .input2 = {1}, .expected = 0});
  // check both ranges empty works
  test<Iter1, Sent1, Iter2, Sent2, 0, 0>({.input1 = {}, .input2 = {}, .expected = 0});
  // the first element is checked properly
  test<Iter1, Sent1, Iter2, Sent2, 5, 2>({.input1 = {5, 4, 3, 2, 1}, .input2 = {1, 5}, .expected = 0});
  // the last element is checked properly
  test<Iter1, Sent1, Iter2, Sent2, 5, 2>({.input1 = {5, 4, 3, 2, 1}, .input2 = {1, 6}, .expected = 4});
  // no match, one-past-the-end iterator should be returned
  test<Iter1, Sent1, Iter2, Sent2, 4, 4>({.input1 = {1, 3, 5, 7}, .input2 = {0, 2, 4, 6}, .expected = 4});
  // input2 contains a single element
  test<Iter1, Sent1, Iter2, Sent2, 4, 1>({.input1 = {1, 3, 5, 7}, .input2 = {1}, .expected = 0});
}

template <class Iter1, class Sent1 = Iter1>
constexpr void test_iterators1() {
  test_iterators<Iter1, Sent1, forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>();
  test_iterators<Iter1, Sent1, forward_iterator<int*>>();
  test_iterators<Iter1, Sent1, bidirectional_iterator<int*>>();
  test_iterators<Iter1, Sent1, random_access_iterator<int*>>();
  test_iterators<Iter1, Sent1, contiguous_iterator<int*>>();
  test_iterators<Iter1, Sent1, int*>();
}

constexpr bool test() {
  test_iterators1<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators1<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_iterators1<forward_iterator<int*>>();
  test_iterators1<bidirectional_iterator<int*>>();
  test_iterators1<random_access_iterator<int*>>();
  test_iterators1<contiguous_iterator<int*>>();
  test_iterators1<int*>();

  { // check that std::ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) ret =
        std::ranges::find_first_of(std::array {1}, std::array {1});
  }

  { // check that the predicate is used
    int a[] = {1, 2, 3, 4};
    int b[] = {2};
    {
      auto ret = std::ranges::find_first_of(std::begin(a), std::end(a),
                                            std::begin(b), std::end(b),
                                            std::ranges::greater{});
      assert(ret == a + 2);
    }
    {
      auto ret = std::ranges::find_first_of(a, b, std::ranges::greater{});
      assert(ret == a + 2);
    }
  }

  { // check that the projections are used
    int a[] = {1, 2, 3, 4};
    int b[] = {4};
    {
      auto ret = std::ranges::find_first_of(std::begin(a), std::end(a),
                                            std::begin(b), std::end(b), {},
                                            [](int i) { return i / 2; },
                                            [](int i) { return i - 3; });
      assert(ret == a + 1);
    }
    {
      auto ret = std::ranges::find_first_of(a, b, {}, [](int i) { return i / 2; }, [](int i) { return i - 3; });
      assert(ret == a + 1);
    }
  }

  { // check that std::invoke is used
    struct S1 {
      constexpr S1(int i_) : i(i_) {}
      constexpr bool compare(int j) const { return j == i; }
      constexpr const S1& identity() const { return *this; }
      int i;
    };
    struct S2 {
      constexpr S2(int i_) : i(i_) {}
      int i;
    };

    {
      S1 a[] = {1, 2, 3, 4};
      S2 b[] = {2, 3};
      auto ret = std::ranges::find_first_of(std::begin(a), std::end(a),
                                            std::begin(b), std::end(b), &S1::compare, &S1::identity, &S2::i);
      assert(ret == a + 1);
    }
    {
      S1 a[] = {1, 2, 3, 4};
      S2 b[] = {2, 3};
      auto ret = std::ranges::find_first_of(a, b, &S1::compare, &S1::identity, &S2::i);
      assert(ret == a + 1);
    }
  }

  { // check that the implicit conversion to bool works
    StrictComparable<int> a[] = {1, 2, 3, 4};
    StrictComparable<int> b[] = {2, 3};
    {
      auto ret = std::ranges::find_first_of(a, std::end(a), b, std::end(b));
      assert(ret == a + 1);
    }
    {
      auto ret = std::ranges::find_first_of(a, b);
      assert(ret == a + 1);
    }
  }

  { // check that the complexity requirements are met
    int a[] = {1, 2, 3, 4};
    int b[] = {2, 3};
    {
      int predCount = 0;
      auto predCounter = [&](int, int) { ++predCount; return false; };
      int proj1Count = 0;
      auto proj1Counter = [&](int i) { ++proj1Count; return i; };
      int proj2Count = 0;
      auto proj2Counter = [&](int i) { ++proj2Count; return i; };
      auto ret = std::ranges::find_first_of(std::begin(a), std::end(a),
                                            std::begin(b), std::end(b), predCounter, proj1Counter, proj2Counter);
      assert(ret == a + 4);
      assert(predCount <= 8);
      assert(proj1Count <= 8);
      assert(proj2Count <= 8);
    }
    {
      int predCount = 0;
      auto predCounter = [&](int, int) { ++predCount; return false; };
      int proj1Count = 0;
      auto proj1Counter = [&](int i) { ++proj1Count; return i; };
      int proj2Count = 0;
      auto proj2Counter = [&](int i) { ++proj2Count; return i; };
      auto ret = std::ranges::find_first_of(a, b, predCounter, proj1Counter, proj2Counter);
      assert(ret == a + 4);
      assert(predCount == 8);
      assert(proj1Count == 8);
      assert(proj2Count == 8);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
