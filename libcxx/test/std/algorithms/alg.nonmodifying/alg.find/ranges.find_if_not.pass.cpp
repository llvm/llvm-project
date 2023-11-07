//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   constexpr I ranges::find_if_not(I first, S last, Pred pred, Proj proj = {});
// template<input_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   constexpr borrowed_iterator_t<R>
//     ranges::find_if_not(R&& r, Pred pred, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct Predicate {
  bool operator()(int);
};

template <class It, class Sent = It>
concept HasFindIfNotIt = requires(It it, Sent sent) { std::ranges::find_if_not(it, sent, Predicate{}); };
static_assert(HasFindIfNotIt<int*>);
static_assert(!HasFindIfNotIt<InputIteratorNotDerivedFrom>);
static_assert(!HasFindIfNotIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasFindIfNotIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasFindIfNotIt<cpp20_input_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasFindIfNotIt<cpp20_input_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!HasFindIfNotIt<int*, int>);
static_assert(!HasFindIfNotIt<int, int*>);

template <class Pred>
concept HasFindIfNotPred = requires(int* it, Pred pred) {std::ranges::find_if_not(it, it, pred); };

static_assert(!HasFindIfNotPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasFindIfNotPred<IndirectUnaryPredicateNotPredicate>);

template <class R>
concept HasFindIfNotR = requires(R r) { std::ranges::find_if_not(r, Predicate{}); };
static_assert(HasFindIfNotR<std::array<int, 0>>);
static_assert(!HasFindIfNotR<int>);
static_assert(!HasFindIfNotR<InputRangeNotDerivedFrom>);
static_assert(!HasFindIfNotR<InputRangeNotIndirectlyReadable>);
static_assert(!HasFindIfNotR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasFindIfNotR<InputRangeNotSentinelSemiregular>);
static_assert(!HasFindIfNotR<InputRangeNotSentinelEqualityComparableWith>);

template <class It, class Sent = It>
constexpr void test_iterators() {
  {
    int a[] = {1, 2, 3, 4};
    std::same_as<It> auto ret = std::ranges::find_if_not(It(a), Sent(It(a + 4)), [c = 0](int) mutable { return c++ <= 2; });
    assert(base(ret) == a + 3);
    assert(*ret == 4);
  }
  {
    int a[] = {1, 2, 3, 4};
    auto range = std::ranges::subrange(It(a), Sent(It(a + 4)));
    std::same_as<It> auto ret = std::ranges::find_if_not(range, [c = 0](int) mutable { return c++ <= 2; });
    assert(base(ret) == a + 3);
    assert(*ret == 4);
  }
}

struct NonConstComparableLValue {
  friend constexpr bool operator==(const NonConstComparableLValue&, const NonConstComparableLValue&) { return false; }
  friend constexpr bool operator==(NonConstComparableLValue&, NonConstComparableLValue&) { return false; }
  friend constexpr bool operator==(const NonConstComparableLValue&, NonConstComparableLValue&) { return false; }
  friend constexpr bool operator==(NonConstComparableLValue&, const NonConstComparableLValue&) { return true; }
};

constexpr bool test() {
  test_iterators<int*>();
  test_iterators<const int*>();
  test_iterators<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();

  { // check that projections are used properly and that they are called with the iterator directly
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::find_if_not(a, a + 4, [&](int* i) { return i != a + 3; }, [](int& i) { return &i; });
      assert(ret == a + 3);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::find_if_not(a, [&](int* i) { return i != a + 3; }, [](int& i) { return &i; });
      assert(ret == a + 3);
    }
  }

  {
    // check that the first element is returned
    {
      struct S {
        int comp;
        int other;
      };
      S a[] = { {0, 0}, {0, 2}, {0, 1} };
      auto ret = std::ranges::find_if_not(a, [](int i){ return i != 0; }, &S::comp);
      assert(ret == a);
      assert(ret->comp == 0);
      assert(ret->other == 0);
    }
    {
      struct S {
        int comp;
        int other;
      };
      S a[] = { {0, 0}, {0, 2}, {0, 1} };
      auto ret = std::ranges::find_if_not(a, a + 3, [](int i) { return i != 0; }, &S::comp);
      assert(ret == a);
      assert(ret->comp == 0);
      assert(ret->other == 0);
    }
  }

  {
    // check that end + 1 iterator is returned with no match
    {
      int a[] = {1, 1, 1};
      auto ret = std::ranges::find_if(a, a + 3, [](int) { return false; });
      assert(ret == a + 3);
    }
    {
      int a[] = {1, 1, 1};
      auto ret = std::ranges::find_if(a, [](int){ return false; });
      assert(ret == a + 3);
    }
  }

  { // check that ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> auto ret =
      std::ranges::find_if_not(std::array{1, 2}, [](int){ return true; });
  }

  { // check that an iterator is returned with a borrowing range
    int a[] = {1, 2, 3, 4};
    std::same_as<int*> auto ret = std::ranges::find_if_not(std::views::all(a), [](int){ return false; });
    assert(ret == a);
    assert(*ret == 1);
  }

  { // check that std::invoke is used
    struct S { int i; };
    S a[] = { S{1}, S{3}, S{2} };
    std::same_as<S*> auto ret = std::ranges::find_if_not(a, [](int) { return true; }, &S::i);
    assert(ret == a + 3);
  }

  { // count projection and predicate invocation count
    {
      int a[] = {1, 2, 3, 4};
      int predicate_count = 0;
      int projection_count = 0;
      auto ret = std::ranges::find_if_not(a, a + 4,
                                      [&](int i) { ++predicate_count; return i != 2; },
                                      [&](int i) { ++projection_count; return i; });
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(predicate_count == 2);
      assert(projection_count == 2);
    }
    {
      int a[] = {1, 2, 3, 4};
      int predicate_count = 0;
      int projection_count = 0;
      auto ret = std::ranges::find_if_not(a,
                                      [&](int i) { ++predicate_count; return i != 2; },
                                      [&](int i) { ++projection_count; return i; });
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(predicate_count == 2);
      assert(projection_count == 2);
    }
  }

  { // check that the return type of `iter::operator*` doesn't change
    {
      NonConstComparableLValue a[] = { NonConstComparableLValue{} };
      auto ret = std::ranges::find_if_not(a, a + 1, [](auto&& e) { return e != NonConstComparableLValue{}; });
      assert(ret == a);
    }
    {
      NonConstComparableLValue a[] = { NonConstComparableLValue{} };
      auto ret = std::ranges::find_if_not(a, [](auto&& e) { return e != NonConstComparableLValue{}; });
      assert(ret == a);
    }
  }

  {
    // check that an empty range works
    {
      std::array<int ,0> a = {};
      auto ret = std::ranges::find_if_not(a.begin(), a.end(), [](int) { return true; });
      assert(ret == a.begin());
    }
    {
      std::array<int, 0> a = {};
      auto ret = std::ranges::find_if_not(a, [](int) { return true; });
      assert(ret == a.begin());
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
