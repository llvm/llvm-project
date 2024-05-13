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
//   constexpr iter_difference_t<I>
//     ranges::count_if(I first, S last, Pred pred, Proj proj = {});
// template<input_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   constexpr range_difference_t<R>
//     ranges::count_if(R&& r, Pred pred, Proj proj = {});

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
concept HasCountIfIt = requires(It it, Sent sent) { std::ranges::count_if(it, sent, Predicate{}); };
static_assert(HasCountIfIt<int*>);
static_assert(!HasCountIfIt<InputIteratorNotDerivedFrom>);
static_assert(!HasCountIfIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasCountIfIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasCountIfIt<cpp20_input_iterator<int*>, SentinelForNotSemiregular>);
static_assert(!HasCountIfIt<cpp20_input_iterator<int*>, InputRangeNotSentinelEqualityComparableWith>);

static_assert(!HasCountIfIt<int*, int>);
static_assert(!HasCountIfIt<int, int*>);

template <class Pred>
concept HasCountIfPred = requires(int* it, Pred pred) {std::ranges::count_if(it, it, pred); };

static_assert(!HasCountIfPred<IndirectUnaryPredicateNotCopyConstructible>);
static_assert(!HasCountIfPred<IndirectUnaryPredicateNotPredicate>);

template <class R>
concept HasCountIfR = requires(R r) { std::ranges::count_if(r, Predicate{}); };
static_assert(HasCountIfR<std::array<int, 0>>);
static_assert(!HasCountIfR<int>);
static_assert(!HasCountIfR<InputRangeNotDerivedFrom>);
static_assert(!HasCountIfR<InputRangeNotIndirectlyReadable>);
static_assert(!HasCountIfR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasCountIfR<InputRangeNotSentinelSemiregular>);
static_assert(!HasCountIfR<InputRangeNotSentinelEqualityComparableWith>);

template <class It, class Sent = It>
constexpr void test_iterators() {
  {
    // simple test
    {
      int a[] = {1, 2, 3, 4};
      std::same_as<std::ptrdiff_t> auto ret =
        std::ranges::count_if(It(a), Sent(It(a + 4)), [](int x) { return x == 4; });
      assert(ret == 1);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto range = std::ranges::subrange(It(a), Sent(It(a + 4)));
      std::same_as<std::ptrdiff_t> auto ret =
        std::ranges::count_if(range, [](int x) { return x == 4; });
      assert(ret == 1);
    }
  }

  {
    // check that an empty range works
    {
      std::array<int, 0> a = {};
      auto ret = std::ranges::count_if(It(a.data()), Sent(It(a.data() + a.size())), [](int) { return true; });
      assert(ret == 0);
    }
    {
      std::array<int, 0> a = {};
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
      auto ret = std::ranges::count_if(range, [](int) { return true; });
      assert(ret == 0);
    }
  }

  {
    // check that a range with a single element works
    {
      std::array a = {2};
      auto ret = std::ranges::count_if(It(a.data()), Sent(It(a.data() + a.size())), [](int i) { return i == 2; });
      assert(ret == 1);
    }
    {
      std::array a = {2};
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
      auto ret = std::ranges::count_if(range, [](int i) { return i == 2; });
      assert(ret == 1);
    }
  }

  {
    // check that 0 is returned with no match
    {
      int a[] = {1, 1, 1};
      auto ret = std::ranges::count_if(It(a), Sent(It(a + 3)), [](int) { return false; });
      assert(ret == 0);
    }
    {
      int a[] = {1, 1, 1};
      auto range = std::ranges::subrange(It(a), Sent(It(a + 3)));
      auto ret = std::ranges::count_if(range, [](int){ return false; });
      assert(ret == 0);
    }
  }

  {
    // check that more than one element is counted
    {
      std::array a = {3, 3, 4, 3, 3};
      auto ret = std::ranges::count_if(It(a.data()), Sent(It(a.data() + a.size())), [](int i) { return i == 3; });
      assert(ret == 4);
    }
    {
      std::array a = {3, 3, 4, 3, 3};
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
      auto ret = std::ranges::count_if(range, [](int i) { return i == 3; });
      assert(ret == 4);
    }
  }

  {
    // check that all elements are counted
    {
      std::array a = {5, 5, 5, 5};
      auto ret = std::ranges::count_if(It(a.data()), Sent(It(a.data() + a.size())), [](int) { return true; });
      assert(ret == 4);
    }
    {
      std::array a = {5, 5, 5, 5};
      auto range = std::ranges::subrange(It(a.data()), Sent(It(a.data() + a.size())));
      auto ret = std::ranges::count_if(range, [](int) { return true; });
      assert(ret == 4);
    }
  }
}

constexpr bool test() {
  test_iterators<int*>();
  test_iterators<const int*>();
  test_iterators<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();

  {
    // check that projections are used properly and that they are called with the iterator directly
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::count_if(a, a + 4, [&](int* i) { return i == a + 3; }, [](int& i) { return &i; });
      assert(ret == 1);
    }
    {
      int a[] = {1, 2, 3, 4};
      auto ret = std::ranges::count_if(a, [&](int* i) { return i == a + 3; }, [](int& i) { return &i; });
      assert(ret == 1);
    }
  }

  {
    // check that std::invoke is used
    {
      struct S {
        int comp;
        int other;
      };
      S a[] = { {0, 0}, {0, 2}, {0, 1} };
      auto ret = std::ranges::count_if(a, [](int i){ return i == 0; }, &S::comp);
      assert(ret == 3);
    }
    {
      struct S {
        int comp;
        int other;
      };
      S a[] = { {0, 0}, {0, 2}, {0, 1} };
      auto ret = std::ranges::count_if(a, a + 3, [](int i) { return i == 0; }, &S::comp);
      assert(ret == 3);
    }
  }

  {
    // check projection and predicate invocation count
    {
      int a[] = {1, 2, 3, 4};
      int predicate_count = 0;
      int projection_count = 0;
      auto ret = std::ranges::count_if(a, a + 4,
                                       [&](int i) { ++predicate_count; return i == 2; },
                                       [&](int i) { ++projection_count; return i; });
      assert(ret == 1);
      assert(predicate_count == 4);
      assert(projection_count == 4);
    }
    {
      int a[] = {1, 2, 3, 4};
      int predicate_count = 0;
      int projection_count = 0;
      auto ret = std::ranges::count_if(a,
                                       [&](int i) { ++predicate_count; return i == 2; },
                                       [&](int i) { ++projection_count; return i; });
      assert(ret == 1);
      assert(predicate_count == 4);
      assert(projection_count == 4);
    }
  }

  {
    // check that an immobile type works
    struct NonMovable {
      NonMovable(const NonMovable&) = delete;
      NonMovable(NonMovable&&) = delete;
      constexpr NonMovable(int i_) : i(i_) {}
      int i;

      bool operator==(const NonMovable&) const = default;
    };
    {
      NonMovable a[] = {9, 8, 4, 3};
      auto ret = std::ranges::count_if(a, a + 4, [](const NonMovable& i) { return i == NonMovable(8); });
      assert(ret == 1);
    }
    {
      NonMovable a[] = {9, 8, 4, 3};
      auto ret = std::ranges::count_if(a, [](const NonMovable& i) { return i == NonMovable(8); });
      assert(ret == 1);
    }
  }

  {
    // check that difference_type is used
    struct DiffTypeIterator {
      using difference_type = signed char;
      using value_type = int;

      int* it = nullptr;

      constexpr DiffTypeIterator() = default;
      constexpr DiffTypeIterator(int* i) : it(i) {}

      constexpr int& operator*() const { return *it; }
      constexpr DiffTypeIterator& operator++() { ++it; return *this; }
      constexpr void operator++(int) { ++it; }

      bool operator==(const DiffTypeIterator&) const = default;
    };

    {
      int a[] = {5, 5, 4, 3, 2, 1};
      std::same_as<signed char> auto ret =
          std::ranges::count_if(DiffTypeIterator(a), DiffTypeIterator(a + 6), [](int& i) { return i == 4; });
      assert(ret == 1);
    }
    {
      int a[] = {5, 5, 4, 3, 2, 1};
      auto range = std::ranges::subrange(DiffTypeIterator(a), DiffTypeIterator(a + 6));
      std::same_as<signed char> auto ret = std::ranges::count_if(range, [](int& i) { return i == 4; });
      assert(ret == 1);
    }
  }

  {
    // check that the predicate can take the argument by lvalue ref
    {
      int a[] = {9, 8, 4, 3};
      auto ret = std::ranges::count_if(a, a + 4, [](int& i) { return i == 8; });
      assert(ret == 1);
    }
    {
      int a[] = {9, 8, 4, 3};
      auto ret = std::ranges::count_if(a, [](int& i) { return i == 8; });
      assert(ret == 1);
    }
  }

  {
    // check that the predicate isn't made const
    struct MutablePredicate {
      constexpr bool operator()(int i) & { return i == 8; }
      constexpr bool operator()(int i) && { return i == 8; }
    };
    {
      int a[] = {9, 8, 4, 3};
      auto ret = std::ranges::count_if(a, a + 4, MutablePredicate{});
      assert(ret == 1);
    }
    {
      int a[] = {9, 8, 4, 3};
      auto ret = std::ranges::count_if(a, MutablePredicate{});
      assert(ret == 1);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
