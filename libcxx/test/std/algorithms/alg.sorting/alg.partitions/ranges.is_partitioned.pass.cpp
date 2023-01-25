//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<input_iterator I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   constexpr bool ranges::is_partitioned(I first, S last, Pred pred, Proj proj = {});
// template<input_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   constexpr bool ranges::is_partitioned(R&& r, Pred pred, Proj proj = {});


#include <algorithm>
#include <cassert>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct Functor {
  bool operator()(auto&&);
};

template <class Iter, class Sent = sentinel_wrapper<Iter>>
concept HasIsPartitionedIt = requires(Iter iter, Sent sent) {
  std::ranges::is_partitioned(iter, sent, Functor{});
};

static_assert(HasIsPartitionedIt<int*>);
static_assert(!HasIsPartitionedIt<InputIteratorNotDerivedFrom>);
static_assert(!HasIsPartitionedIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasIsPartitionedIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasIsPartitionedIt<int*, SentinelForNotSemiregular>);
static_assert(!HasIsPartitionedIt<int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Pred>
concept HasIsPartitionedItPred = requires(int* first, int* last, Pred pred) {
  std::ranges::is_partitioned(first, last, pred);
};

static_assert(HasIsPartitionedItPred<Functor>);
static_assert(!HasIsPartitionedItPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasIsPartitionedItPred<IndirectUnaryPredicateNotPredicate>);

template <class Range>
concept HasIsPartitionedR = requires (Range range) {
    std::ranges::is_partitioned(range, Functor{});
};

static_assert(HasIsPartitionedR<UncheckedRange<int*>>);
static_assert(!HasIsPartitionedR<InputRangeNotDerivedFrom>);
static_assert(!HasIsPartitionedR<InputRangeNotIndirectlyReadable>);
static_assert(!HasIsPartitionedR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasIsPartitionedR<InputRangeNotSentinelSemiregular>);
static_assert(!HasIsPartitionedR<InputRangeNotSentinelEqualityComparableWith>);

template <class Pred>
concept HasIsPartitionedRPred = requires(Pred pred) {
  std::ranges::is_partitioned(UncheckedRange<int*>{}, pred);
};

static_assert(HasIsPartitionedRPred<Functor>);
static_assert(!HasIsPartitionedRPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasIsPartitionedRPred<IndirectUnaryPredicateNotPredicate>);

template <class Iter, class Sent = Iter>
constexpr void test_iterators() {
  { // simple test
    {
      int a[] = {1, 2, 3, 4, 5};
      std::same_as<bool> decltype(auto) ret =
          std::ranges::is_partitioned(Iter(a), Sent(Iter(a + 5)), [](int i) { return i < 3; });
      assert(ret);
    }
    {
      int a[] = {1, 2, 3, 4, 5};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
      std::same_as<bool> decltype(auto) ret = std::ranges::is_partitioned(range, [](int i) { return i < 3; });
      assert(ret);
    }
  }

  { // check that it's partitoned if the predicate is true for all elements
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::is_partitioned(Iter(a), Sent(Iter(a + 4)), [](int) { return true; });
      assert(ret);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 4)));
      auto ret = std::ranges::is_partitioned(range, [](int) { return true; });
      assert(ret);
    }
  }

  { // check that it's partitioned if the predicate is false for all elements
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::is_partitioned(Iter(a), Sent(Iter(a + 4)), [](int) { return false; });
      assert(ret);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 4)));
      auto ret = std::ranges::is_partitioned(range, [](int) { return false; });
      assert(ret);
    }
  }

  { // check that false is returned if the range isn't partitioned
    {
      int a[] = {1, 3, 2, 4};
      auto ret = std::ranges::is_partitioned(Iter(a), Sent(Iter(a + 4)), [](int i) { return i < 3; });
      assert(!ret);
    }
    {
      int a[] = {1, 3, 2, 4};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 4)));
      auto ret = std::ranges::is_partitioned(range, [](int i) { return i < 3; });
      assert(!ret);
    }
  }

  { // check that an empty range is partitioned
    {
      int a[] = {};
      auto ret = std::ranges::is_partitioned(Iter(a), Sent(Iter(a)), [](int i) { return i < 3; });
      assert(ret);
    }
    {
      int a[] = {};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a)));
      auto ret = std::ranges::is_partitioned(range, [](int i) { return i < 3; });
      assert(ret);
    }
  }

  { // check that a single element is partitioned
    {
      int a[] = {1};
      auto ret = std::ranges::is_partitioned(Iter(a), Sent(Iter(a + 1)), [](int i) { return i < 3; });
      assert(ret);
    }
    {
      int a[] = {1};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 1)));
      auto ret = std::ranges::is_partitioned(range, [](int i) { return i < 3; });
      assert(ret);
    }
  }

  { // check that it is partitioned when the first element is the partition point
    {
      int a[] = {0, 1, 1};
      auto ret = std::ranges::is_partitioned(Iter(a), Sent(Iter(a + 3)), [](int i) { return i < 1; });
      assert(ret);
    }
    {
      int a[] = {0, 1, 1};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 3)));
      auto ret = std::ranges::is_partitioned(range, [](int i) { return i < 1; });
      assert(ret);
    }
  }

  { // check that it is partitioned when the last element is the partition point
    {
      int a[] = {0, 0, 1};
      auto ret = std::ranges::is_partitioned(Iter(a), Sent(Iter(a + 3)), [](int i) { return i < 1; });
      assert(ret);
    }
    {
      int a[] = {0, 0, 1};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 3)));
      auto ret = std::ranges::is_partitioned(range, [](int i) { return i < 1; });
      assert(ret);
    }
  }
}

constexpr bool test() {
  test_iterators<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_iterators<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();
  test_iterators<int*>();
  test_iterators<const int*>();

  { // check that std:invoke is used
    struct S {
      int check;
      int other;

      constexpr S& identity() {
        return *this;
      }
    };
    {
      S a[] = {{1, 2}, {3, 4}, {5, 6}};
      auto ret = std::ranges::is_partitioned(a, a + 3, &S::check, &S::identity);
      assert(ret);
    }
    {
      S a[] = {{1, 2}, {3, 4}, {5, 6}};
      auto ret = std::ranges::is_partitioned(a, &S::check, &S::identity);
      assert(ret);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
