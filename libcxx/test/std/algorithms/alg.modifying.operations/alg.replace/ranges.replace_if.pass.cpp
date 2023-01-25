//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_iterator I, sentinel_for<I> S, class T, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   requires indirectly_writable<I, const T&>
//   constexpr I ranges::replace_if(I first, S last, Pred pred, const T& new_value, Proj proj = {});
// template<input_range R, class T, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   requires indirectly_writable<iterator_t<R>, const T&>
//   constexpr borrowed_iterator_t<R>
//     ranges::replace_if(R&& r, Pred pred, const T& new_value, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

struct FalsePredicate {
  bool operator()(auto&&) { return false; }
};

template <class Iter, class Sent = sentinel_wrapper<Iter>>
concept HasReplaceIt = requires(Iter iter, Sent sent) { std::ranges::replace_if(iter, sent, FalsePredicate{}, 0); };

static_assert(HasReplaceIt<int*>);
static_assert(!HasReplaceIt<InputIteratorNotDerivedFrom>);
static_assert(!HasReplaceIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasReplaceIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasReplaceIt<int*, SentinelForNotSemiregular>);
static_assert(!HasReplaceIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasReplaceIt<int**>); // not indirectly_writable
static_assert(!HasReplaceIt<IndirectBinaryPredicateNotIndirectlyReadable>);

template <class Range>
concept HasReplaceR = requires(Range range) { std::ranges::replace_if(range, FalsePredicate{}, 0); };

static_assert(HasReplaceR<UncheckedRange<int*>>);
static_assert(!HasReplaceR<InputRangeNotDerivedFrom>);
static_assert(!HasReplaceR<InputRangeNotIndirectlyReadable>);
static_assert(!HasReplaceR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasReplaceR<InputRangeNotSentinelSemiregular>);
static_assert(!HasReplaceR<InputRangeNotSentinelEqualityComparableWith>);
static_assert(!HasReplaceR<UncheckedRange<int**>>); // not indirectly_writable
static_assert(!HasReplaceR<InputRangeIndirectBinaryPredicateNotIndirectlyReadable>);

template <class Iter, class Sent, int N, class Pred>
constexpr void test(std::array<int, N> a_, Pred pred, int val, std::array<int, N> expected) {
  {
    auto a = a_;
    std::same_as<Iter> auto ret = std::ranges::replace_if(Iter(a.data()), Sent(Iter(a.data() + N)),
                                                          pred,
                                                          val);
    assert(base(ret) == a.data() + N);
    assert(a == expected);
  }
  {
    auto a = a_;
    auto range = std::ranges::subrange(Iter(a.data()), Sent(Iter(a.data() + N)));
    std::same_as<Iter> auto ret = std::ranges::replace_if(range, pred, val);
    assert(base(ret) == a.data() + N);
    assert(a == expected);
  }
}

template <class Iter, class Sent = Iter>
constexpr void test_iterators() {
  // simple test
  test<Iter, Sent, 4>({1, 2, 3, 4}, [](int i) { return i < 3; }, 23, {23, 23, 3, 4});
  // no match
  test<Iter, Sent, 4>({1, 2, 3, 4}, [](int i) { return i < 0; }, 23, {1, 2, 3, 4});
  // all match
  test<Iter, Sent, 4>({1, 2, 3, 4}, [](int i) { return i > 0; }, 23, {23, 23, 23, 23});
  // empty range
  test<Iter, Sent, 0>({}, [](int i) { return i > 0; }, 23, {});
  // single element range
  test<Iter, Sent, 1>({1}, [](int i) { return i > 0; }, 2, {2});
}

constexpr bool test() {
  test_iterators<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_iterators<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();
  test_iterators<int*>();

  { // check that the projection is used
    struct S {
      constexpr S(int i_) : i(i_) {}
      int i;
    };
    {
      S a[] = {1, 2, 3, 4};
      std::ranges::replace_if(a, a + 4, [](int i) { return i == 3; }, S{0}, &S::i);
    }
    {
      S a[] = {1, 2, 3, 4};
      std::ranges::replace_if(a, [](int i) { return i == 3; }, S{0}, &S::i);
    }
  }

  { // check that std::invoke is used
    struct S {
      constexpr S(int i_) : i(i_) {}
      constexpr bool check() const { return false; }
      constexpr const S& identity() const { return *this; }
      int i;
    };
    {
      S a[] = {1, 2, 3, 4};
      auto ret = std::ranges::replace_if(std::begin(a), std::end(a), &S::check, S{2}, &S::identity);
      assert(ret == std::end(a));
    }
    {
      S a[] = {1, 2, 3, 4};
      auto ret = std::ranges::replace_if(a, &S::check, S{2}, &S::identity);
      assert(ret == std::end(a));
    }
  }

  { // check that the implicit conversion to bool works
    {
      int a[] = {1, 2, 2, 4};
      auto ret = std::ranges::replace_if(std::begin(a), std::end(a), [](int) { return BooleanTestable{false}; }, 2);
      assert(ret == std::end(a));
    }
    {
      int a[] = {1, 2, 2, 4};
      auto ret = std::ranges::replace_if(a, [](int) { return BooleanTestable{false}; }, 2);
      assert(ret == std::end(a));
    }
  }

  { // check that std::ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) ret =
        std::ranges::replace_if(std::array {1, 2, 3, 4}, [](int) { return false; }, 1);
  }

  { // check that the complexity requirements are met
    {
      int predicateCount = 0;
      auto pred = [&](int) { ++predicateCount; return false; };
      auto projectionCount = 0;
      auto proj = [&](int i) { ++projectionCount; return i; };
      int a[] = {1, 2, 3, 4, 5};
      auto ret = std::ranges::replace_if(a, a + 5, pred, 1, proj);
      assert(ret == a + 5);
      assert(predicateCount == 5);
      assert(projectionCount == 5);
    }
    {
      int predicateCount = 0;
      auto pred = [&](int) { ++predicateCount; return false; };
      auto projectionCount = 0;
      auto proj = [&](int i) { ++projectionCount; return i; };
      int a[] = {1, 2, 3, 4, 5};
      auto ret = std::ranges::replace_if(a, pred, 1, proj);
      assert(ret == a + 5);
      assert(predicateCount == 5);
      assert(projectionCount == 5);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
