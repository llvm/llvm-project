//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<forward_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   constexpr subrange<I> ranges::find_last_if(I first, S last, Pred pred, Proj proj = {});
// template<forward_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   constexpr borrowed_subrange_t<R> ranges::find_last_if(R&& r, Pred pred, Proj proj = {});

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
concept HasFindLastIfIt = requires(It it, Sent sent) { std::ranges::find_last_if(it, sent, Predicate{}); };
static_assert(HasFindLastIfIt<int*>);
static_assert(HasFindLastIfIt<forward_iterator<int*>>);
static_assert(!HasFindLastIfIt<cpp20_input_iterator<int*>>);
static_assert(!HasFindLastIfIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasFindLastIfIt<ForwardIteratorNotIncrementable>);
static_assert(!HasFindLastIfIt<forward_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasFindLastIfIt<forward_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!HasFindLastIfIt<int*, int>);
static_assert(!HasFindLastIfIt<int, int*>);

template <class Pred>
concept HasFindLastIfPred = requires(int* it, Pred pred) { std::ranges::find_last_if(it, it, pred); };

static_assert(!HasFindLastIfPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasFindLastIfPred<IndirectUnaryPredicateNotPredicate>);

template <class R>
concept HasFindLastIfR = requires(R r) { std::ranges::find_last_if(r, Predicate{}); };
static_assert(HasFindLastIfR<std::array<int, 0>>);
static_assert(!HasFindLastIfR<int>);
static_assert(!HasFindLastIfR<ForwardRangeNotDerivedFrom>);
static_assert(!HasFindLastIfR<ForwardRangeNotIncrementable>);
static_assert(!HasFindLastIfR<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasFindLastIfR<ForwardRangeNotSentinelEqualityComparableWith>);

template <class It, class Sent = It>
constexpr void test_iterators() {
  {
    int a[] = {1, 2, 3, 4};
    std::same_as<std::ranges::subrange<It>> auto ret =
        std::ranges::find_last_if(It(a), Sent(It(a + 4)), [](int x) { return x == 4; });
    assert(base(ret.begin()) == a + 3);
    assert(*ret.begin() == 4);
  }
  {
    int a[]    = {1, 2, 3, 4};
    auto range = std::ranges::subrange(It(a), Sent(It(a + 4)));

    std::same_as<std::ranges::subrange<It>> auto ret = std::ranges::find_last_if(range, [](int x) { return x == 4; });
    assert(base(ret.begin()) == a + 3);
    assert(*ret.begin() == 4);
  }
}

struct NonConstComparable {
  friend constexpr bool operator==(const NonConstComparable&, const NonConstComparable&) { return false; }
  friend constexpr bool operator==(NonConstComparable&, NonConstComparable&) { return false; }
  friend constexpr bool operator==(const NonConstComparable&, NonConstComparable&) { return false; }
  friend constexpr bool operator==(NonConstComparable&, const NonConstComparable&) { return true; }
};

constexpr bool test() {
  test_iterators<int*>();
  test_iterators<const int*>();
  test_iterators<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();

  {
    // check that projections are used properly and that they are called with the iterator directly
    {
      int a[] = {1, 2, 3, 4};
      auto ret =
          std::ranges::find_last_if(a, a + 4, [&](int* i) { return i == a + 3; }, [](int& i) { return &i; }).begin();
      assert(ret == a + 3);
    }
    {
      int a[]  = {1, 2, 3, 4};
      auto ret = std::ranges::find_last_if(a, [&](int* i) { return i == a + 3; }, [](int& i) { return &i; }).begin();
      assert(ret == a + 3);
    }
  }

  {
    // check that the last element is returned
    {
      struct S {
        int comp;
        int other;
      };
      S a[]    = {{0, 0}, {0, 2}, {0, 1}};
      auto ret = std::ranges::find_last_if(a, [](int i) { return i == 0; }, &S::comp).begin();
      assert(ret == a + 2);
      assert(ret->comp == 0);
      assert(ret->other == 1);
    }
    {
      struct S {
        int comp;
        int other;
      };
      S a[]    = {{0, 0}, {0, 2}, {0, 1}};
      auto ret = std::ranges::find_last_if(a, a + 3, [](int i) { return i == 0; }, &S::comp).begin();
      assert(ret == a + 2);
      assert(ret->comp == 0);
      assert(ret->other == 1);
    }
  }

  {
    // check that end iterator is returned with no match
    {
      int a[]  = {1, 1, 1};
      auto ret = std::ranges::find_last_if(a, a + 3, [](int) { return false; }).begin();
      assert(ret == a + 3);
    }
    {
      int a[]  = {1, 1, 1};
      auto ret = std::ranges::find_last_if(a, [](int) { return false; }).begin();
      assert(ret == a + 3);
    }
  }

  {
    // check that ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> auto ret =
        std::ranges::find_last_if(std::array{1, 2}, [](int) { return false; });
  }

  {
    // check that an iterator is returned with a borrowing range
    int a[] = {1, 2, 3, 4};
    std::same_as<std::ranges::subrange<int*>> auto ret =
        std::ranges::find_last_if(std::views::all(a), [](int) { return true; });
    assert(ret.begin() == a + 3);
    assert(*ret.begin() == 4);
  }

  {
    // check that std::invoke is used
    struct S {
      int i;
    };
    S a[] = {S{1}, S{3}, S{2}};

    std::same_as<S*> auto ret = std::ranges::find_last_if(a, [](int) { return false; }, &S::i).begin();
    assert(ret == a + 3);
  }

  {
    // count projection and predicate invocation count
    {
      int a[]              = {1, 2, 3, 4};
      int predicate_count  = 0;
      int projection_count = 0;
      auto ret =
          std::ranges::find_last_if(
              a,
              a + 4,
              [&](int i) {
                ++predicate_count;
                return i == 2;
              },
              [&](int i) {
                ++projection_count;
                return i;
              })
              .begin();
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(predicate_count == 3);
      assert(projection_count == 3);
    }
    {
      int a[]              = {1, 2, 3, 4};
      int predicate_count  = 0;
      int projection_count = 0;
      auto ret =
          std::ranges::find_last_if(
              a,
              [&](int i) {
                ++predicate_count;
                return i == 2;
              },
              [&](int i) {
                ++projection_count;
                return i;
              })
              .begin();
      assert(ret == a + 1);
      assert(*ret == 2);
      assert(predicate_count == 3);
      assert(projection_count == 3);
    }
  }

  {
    // check that the return type of `iter::operator*` doesn't change
    {
      NonConstComparable a[] = {NonConstComparable{}};

      auto ret = std::ranges::find_last_if(a, a + 1, [](auto&& e) { return e == NonConstComparable{}; }).begin();
      assert(ret == a);
    }
    {
      NonConstComparable a[] = {NonConstComparable{}};

      auto ret = std::ranges::find_last_if(a, [](auto&& e) { return e == NonConstComparable{}; }).begin();
      assert(ret == a);
    }
  }

  {
    // check that an empty range works
    {
      std::array<int, 0> a = {};

      auto ret = std::ranges::find_last_if(a.begin(), a.end(), [](int) { return true; }).begin();
      assert(ret == a.begin());
    }
    {
      std::array<int, 0> a = {};

      auto ret = std::ranges::find_last_if(a, [](int) { return true; }).begin();
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
