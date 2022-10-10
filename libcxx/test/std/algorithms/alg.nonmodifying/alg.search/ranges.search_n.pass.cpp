//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<forward_iterator I, sentinel_for<I> S, class T,
//          class Pred = ranges::equal_to, class Proj = identity>
//   requires indirectly_comparable<I, const T*, Pred, Proj>
//   constexpr subrange<I>
//     ranges::search_n(I first, S last, iter_difference_t<I> count,
//                      const T& value, Pred pred = {}, Proj proj = {});
// template<forward_range R, class T, class Pred = ranges::equal_to,
//          class Proj = identity>
//   requires indirectly_comparable<iterator_t<R>, const T*, Pred, Proj>
//   constexpr borrowed_subrange_t<R>
//     ranges::search_n(R&& r, range_difference_t<R> count,
//                      const T& value, Pred pred = {}, Proj proj = {});

#include <algorithm>
#include <array>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class Iter1, class Sent1 = Iter1>
concept HasSearchNIt = requires (Iter1 first1, Sent1 last1) {
  std::ranges::search_n(first1, last1, 0, 0);
};

static_assert(HasSearchNIt<int*>);
static_assert(!HasSearchNIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasSearchNIt<ForwardIteratorNotIncrementable>);
static_assert(!HasSearchNIt<int*, SentinelForNotSemiregular>);
static_assert(!HasSearchNIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasSearchNIt<int**, int**>); // not indirectly comparable

template <class Range1, class Range2 = UncheckedRange<int*>>
concept HasSearchNR = requires (Range1 range) {
  std::ranges::search_n(range, 0, 0);
};

static_assert(HasSearchNR<UncheckedRange<int*>>);
static_assert(!HasSearchNR<ForwardRangeNotDerivedFrom>);
static_assert(!HasSearchNR<ForwardIteratorNotIncrementable>);
static_assert(!HasSearchNR<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasSearchNR<ForwardRangeNotSentinelEqualityComparableWith>);
static_assert(!HasSearchNR<UncheckedRange<int**>>); // not indirectly comparable

template <class Iter, class Sent = Iter>
constexpr void test_iterators() {
  { // simple test
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      std::same_as<std::ranges::subrange<Iter, Iter>> decltype(auto) ret =
          std::ranges::search_n(Iter(a), Sent(Iter(a + 6)), 1, 3);
      assert(base(ret.begin()) == a + 2);
      assert(base(ret.end()) == a + 3);
    }
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 6)));
      std::same_as<std::ranges::subrange<Iter, Iter>> decltype(auto) ret = std::ranges::search_n(range, 1, 3);
      assert(base(ret.begin()) == a + 2);
      assert(base(ret.end()) == a + 3);
    }
  }

  { // matching part begins at the front
    {
      int a[] = {7, 7, 3, 7, 3, 6};
      auto ret = std::ranges::search_n(Iter(a), Sent(Iter(a + 6)), 2, 7);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 2);
    }
    {
      int a[] = {7, 7, 3, 7, 3, 6};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 6)));
      auto ret = std::ranges::search_n(range, 2, 7);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 2);
    }
  }

  { // matching part ends at the back
    {
      int a[] = {9, 3, 6, 4, 4};
      auto ret = std::ranges::search_n(Iter(a), Sent(Iter(a + 5)), 2, 4);
      assert(base(ret.begin()) == a + 3);
      assert(base(ret.end()) == a + 5);
    }
    {
      int a[] = {9, 3, 6, 4, 4};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
      auto ret = std::ranges::search_n(range, 2, 4);
      assert(base(ret.begin()) == a + 3);
      assert(base(ret.end()) == a + 5);
    }
  }

  { // pattern does not match
    {
      int a[] = {9, 3, 6, 4, 8};
      auto ret = std::ranges::search_n(Iter(a), Sent(Iter(a + 5)), 1, 1);
      assert(base(ret.begin()) == a + 5);
      assert(base(ret.end()) == a + 5);
    }
    {
      int a[] = {9, 3, 6, 4, 8};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 5)));
      auto ret = std::ranges::search_n(range, 1, 1);
      assert(base(ret.begin()) == a + 5);
      assert(base(ret.end()) == a + 5);
    }
  }

  { // range and pattern are identical
    {
      int a[] = {1, 1, 1, 1};
      auto ret = std::ranges::search_n(Iter(a), Sent(Iter(a + 4)), 4, 1);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 4);
    }
    {
      int a[] = {1, 1, 1, 1};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 4)));
      auto ret = std::ranges::search_n(range, 4, 1);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 4);
    }
  }

  { // pattern is longer than range
    {
      int a[] = {3, 3, 3};
      auto ret = std::ranges::search_n(Iter(a), Sent(Iter(a + 3)), 4, 3);
      assert(base(ret.begin()) == a + 3);
      assert(base(ret.end()) == a + 3);
    }
    {
      int a[] = {3, 3, 3};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 3)));
      auto ret = std::ranges::search_n(range, 4, 3);
      assert(base(ret.begin()) == a + 3);
      assert(base(ret.end()) == a + 3);
    }
  }

  { // pattern has zero length
    {
      int a[] = {6, 7, 8};
      auto ret = std::ranges::search_n(Iter(a), Sent(Iter(a + 3)), 0, 7);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a);
    }
    {
      int a[] = {6, 7, 8};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 3)));
      auto ret = std::ranges::search_n(range, 0, 7);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a);
    }
  }

  { // range has zero length
    {
      int a[] = {};
      auto ret = std::ranges::search_n(Iter(a), Sent(Iter(a)), 1, 1);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a);
    }
    {
      int a[] = {};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a)));
      auto ret = std::ranges::search_n(range, 1, 1);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a);
    }
  }

  { // check that the first match is returned
    {
      int a[] = {6, 6, 8, 6, 6, 8, 6, 6, 8};
      auto ret = std::ranges::search_n(Iter(a), Sent(Iter(a + 9)), 2, 6);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 2);
    }
    {
      int a[] = {6, 6, 8, 6, 6, 8, 6, 6, 8};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 9)));
      auto ret = std::ranges::search_n(range, 2, 6);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 2);
    }
  }

  { // check that the predicate is used
    {
      int a[] = {1, 4, 4, 3, 6, 1};
      auto ret = std::ranges::search_n(Iter(a), Sent(Iter(a + 6)),
                                       3,
                                       4,
                                       [](int l, int r) { return l == r || l == 1; });
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 3);
    }
    {
      int a[] = {1, 4, 4, 3, 6, 1};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 6)));
      auto ret = std::ranges::search_n(range, 3, 4, [](int l, int r) { return l == r || l == 1; });
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 3);
    }
  }

  { // check that the projections are used
    {
      int a[] = {1, 3, 1, 6, 5, 6};
      auto ret = std::ranges::search_n(Iter(a), Sent(Iter(a + 6)),
                                       3,
                                       6,
                                       {},
                                       [](int i) { return i % 2 == 0 ? i : i + 1; });
      assert(base(ret.begin()) == a + 3);
      assert(base(ret.end()) == a + 6);
    }
    {
      int a[] = {1, 3, 1, 6, 5, 6};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 6)));
      auto ret = std::ranges::search_n(range,
                                       3,
                                       6,
                                       {},
                                       [](int i) { return i % 2 == 0 ? i : i + 1; });
      assert(base(ret.begin()) == a + 3);
      assert(base(ret.end()) == a + 6);
    }
  }
}
constexpr bool test() {
  test_iterators<forward_iterator<int*>>();
  test_iterators<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<random_access_iterator<int*>, sized_sentinel<random_access_iterator<int*>>>();
  test_iterators<contiguous_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>, sized_sentinel<contiguous_iterator<int*>>>();
  test_iterators<int*>();

  { // check that std::invoke is used
    struct S {
      int i;

      constexpr S identity() {
        return *this;
      }

      constexpr bool compare(int o) {
        return i == o;
      }
    };
    {
      S a[] = {{1}, {2}, {3}, {4}};
      auto ret = std::ranges::search_n(a, a + 4, 1, 2, &S::compare, &S::identity);
      assert(ret.begin() == a + 1);
      assert(ret.end() == a + 2);
    }
    {
      S a[] = {{1}, {2}, {3}, {4}};
      auto ret = std::ranges::search_n(a, 1, 2, &S::compare, &S::identity);
      assert(ret.begin() == a + 1);
      assert(ret.end() == a + 2);
    }
  }

  { // check that std::ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) ret =
        std::ranges::search_n(std::array {1, 2, 3, 4}, 1, 0);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
