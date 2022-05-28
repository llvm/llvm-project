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

// template<input_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   constexpr bool ranges::all_of(I first, S last, Pred pred, Proj proj = {});
// template<input_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   constexpr bool ranges::all_of(R&& r, Pred pred, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct UnaryFunctor {
  bool operator()(auto&&);
};

template <class It, class Sent = sentinel_wrapper<It>>
concept HasAllOfIt = requires(It first, Sent last) { std::ranges::all_of(first, last, UnaryFunctor{}); };

static_assert(HasAllOfIt<int*>);
static_assert(!HasAllOfIt<InputIteratorNotDerivedFrom>);
static_assert(!HasAllOfIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasAllOfIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasAllOfIt<int*, SentinelForNotSemiregular>);
static_assert(!HasAllOfIt<int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Func>
concept HasAllOfItFunc = requires(int* ptr) { std::ranges::all_of(ptr, ptr, Func{}); };

static_assert(HasAllOfItFunc<UnaryFunctor>);
static_assert(!HasAllOfItFunc<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasAllOfItFunc<IndirectUnaryPredicateNotPredicate>);

template <class Range>
concept HasAllOfR = requires(Range range) { std::ranges::all_of(range, UnaryFunctor{}); };

static_assert(HasAllOfR<std::array<int, 10>>);
static_assert(!HasAllOfR<InputRangeNotDerivedFrom>);
static_assert(!HasAllOfR<InputRangeNotIndirectlyReadable>);
static_assert(!HasAllOfR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasAllOfR<InputRangeNotSentinelSemiregular>);
static_assert(!HasAllOfR<InputRangeNotSentinelEqualityComparableWith>);

template <class Func>
concept HasAllOfRFunc = requires(std::array<int, 10> range) { std::ranges::all_of(range, Func{}); };

static_assert(HasAllOfRFunc<UnaryFunctor>);
static_assert(!HasAllOfRFunc<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasAllOfRFunc<IndirectUnaryPredicateNotPredicate>);

template <class It, class Sent = It>
constexpr void test_iterators() {
  { // simple test
    {
      int a[] = {1, 2, 3, 4};
      std::same_as<bool> decltype(auto) ret = std::ranges::all_of(It(a), Sent(It(a + 4)), [](int) { return true; });
      assert(ret);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto range = std::ranges::subrange(It(a), Sent(It(a + 4)));
      std::same_as<bool> decltype(auto) ret = std::ranges::all_of(range, [](int) { return true; });
      assert(ret);
    }
  }

  { // check that an empty range works
    std::array<int, 0> a;
    assert(std::ranges::all_of(It(a.data()), Sent(It(a.data())), [](int) { return false; }));
    auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data())));
    assert(std::ranges::all_of(range, [](int) { return false; }));
  }

  { // check that the complexity requirements are met
    {
      int predicateCount = 0;
      int projectionCount = 0;
      auto pred = [&](int) { ++predicateCount; return true; };
      auto proj = [&](int i) { ++projectionCount; return i; };
      std::array a = {9, 7, 5, 3};
      assert(std::ranges::all_of(It(a.begin()), Sent(It(a.end())), pred, proj));
      assert(predicateCount == 4);
      assert(projectionCount == 4);
    }
    {
      int predicateCount = 0;
      int projectionCount = 0;
      auto pred = [&](int) { ++predicateCount; return true; };
      auto proj = [&](int i) { ++projectionCount; return i; };
      std::array a = {9, 7, 5, 3};
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
      assert(std::ranges::all_of(range, pred, proj));
      assert(predicateCount == 4);
      assert(projectionCount == 4);
    }
  }

  { // check that false is returned if no element satisfies the condition
    std::array a = {1, 2, 3, 4};
    assert(!std::ranges::all_of(It(a.data()), Sent(It(a.data() + a.size())), [](int i) { return i > 5; }));
    auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
    assert(!std::ranges::all_of(range, [](int i) { return i > 5; }));
  }

  { // check that true is returned if all elements satisfy the condition
    std::array a = {1, 2, 3, 4};
    assert(std::ranges::all_of(It(a.data()), Sent(It(a.data() + a.size())), [](int i) { return i < 5; }));
    auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
    assert(std::ranges::all_of(range, [](int i) { return i < 5; }));
  }

  { // check that false is returned if ony one elements satisfies the condition
    std::array a = {1, 2, 3, 4, 6};
    assert(!std::ranges::all_of(It(a.data()), Sent(It(a.data() + a.size())), [](int i) { return i > 5; }));
    auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
    assert(!std::ranges::all_of(range, [](int i) { return i > 5; }));
  }
}

constexpr bool test() {
  test_iterators<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();
  test_iterators<int*>();

  { // check that std::invoke is used
    struct S { int check; int other; };
    S a[] = {{1, 2}, {1, 7}, {1, 3}};
    assert(std::ranges::all_of(a, a + 3, [](int i) { return i == 1; }, &S::check));
    assert(std::ranges::all_of(a, [](int i) { return i == 1; }, &S::check));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
