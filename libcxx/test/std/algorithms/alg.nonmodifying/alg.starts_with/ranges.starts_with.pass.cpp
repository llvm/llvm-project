//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2,
//          class Pred = ranges::equal_to, class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_comparable<I1, I2, Pred, Proj1, Proj2>
//   constexpr bool ranges::starts_with(I1 first1, S1 last1, I2 first2, S2 last2, Pred pred = {},
//                                      Proj1 proj1 = {}, Proj2 proj2 = {});
// template<input_range R1, input_range R2, class Pred = ranges::equal_to, class Proj1 = identity,
//          class Proj2 = identity>
//   requires indirectly_comparable<iterator_t<R1>, iterator_t<R2>, Pred, Proj1, Proj2>
//   constexpr bool ranges::starts_with(R1&& r1, R2&& r2, Pred pred = {},
//                                      Proj1 proj1 = {}, Proj2 proj2 = {});

#include <algorithm>
#include <array>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class Iter1, class Sent1 = Iter1, class Iter2 = int*, class Sent2 = Iter2>
concept HasStartsWithIt = requires(Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2) {
  std::ranges::starts_with(first1, last1, first2, last2);
};

static_assert(HasStartsWithIt<int*>);
static_assert(HasStartsWithIt<ForwardIteratorNotDerivedFrom>);
static_assert(HasStartsWithIt<ForwardIteratorNotIncrementable>);
static_assert(!HasStartsWithIt<int*, SentinelForNotSemiregular>);
static_assert(!HasStartsWithIt<int*, int*, int**>); // not indirectly comparable
static_assert(!HasStartsWithIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(HasStartsWithIt<int*, int*, ForwardIteratorNotDerivedFrom>);
static_assert(HasStartsWithIt<int*, int*, ForwardIteratorNotIncrementable>);
static_assert(!HasStartsWithIt<int*, int*, int*, SentinelForNotSemiregular>);
static_assert(!HasStartsWithIt<int*, int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Range1, class Range2 = UncheckedRange<int*>>
concept HasStartsWithR = requires(Range1 range1, Range2 range2) { std::ranges::starts_with(range1, range2); };

static_assert(HasStartsWithR<UncheckedRange<int*>>);
static_assert(HasStartsWithR<ForwardRangeNotDerivedFrom>);
static_assert(!HasStartsWithR<ForwardIteratorNotIncrementable>);
static_assert(!HasStartsWithR<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasStartsWithR<ForwardRangeNotSentinelEqualityComparableWith>);
static_assert(!HasStartsWithR<UncheckedRange<int*>, UncheckedRange<int**>>); // not indirectly comparable
static_assert(HasStartsWithR<UncheckedRange<int*>, ForwardRangeNotDerivedFrom>);
static_assert(HasStartsWithR<UncheckedRange<int*>, ForwardRangeNotIncrementable>);
static_assert(!HasStartsWithR<UncheckedRange<int*>, ForwardRangeNotSentinelSemiregular>);
static_assert(!HasStartsWithR<UncheckedRange<int*>, ForwardRangeNotSentinelEqualityComparableWith>);

// clang-format off
template <class Iter1, class Sent1 = Iter1, class Iter2, class Sent2 = Iter2>
constexpr void test_iterators() {
  {  // simple tests
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      int p[] = {1, 2};
      std::same_as<bool> decltype(auto) ret =
        std::ranges::starts_with(Iter1(a), Sent1(Iter1(a + 6)), Iter2(p), Sent2(Iter2(p + 2)));
      assert(ret);
    }
    {
      int a[]                               = {1, 2, 3, 4, 5, 6};
      int p[]                               = {1, 2};
      auto whole                            = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
      auto prefix                           = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 2)));
      std::same_as<bool> decltype(auto) ret = std::ranges::starts_with(whole, prefix);
      assert(ret);
    }
  }

  { // prefix doesn't match
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      int p[] = {4, 5, 6};
      std::same_as<bool> decltype(auto) ret =
          std::ranges::starts_with(Iter1(a), Sent1(Iter1(a + 6)), Iter2(p), Sent2(Iter2(p + 3)));
      assert(!ret);
    }
    {
      int a[]                               = {1, 2, 3, 4, 5, 6};
      int p[]                               = {4, 5, 6};
      auto whole                            = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
      auto prefix                           = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
      std::same_as<bool> decltype(auto) ret = std::ranges::starts_with(whole, prefix);
      assert(!ret);
    }
  }

  { // range and prefix are identical
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      int p[] = {1, 2, 3, 4, 5, 6};
      std::same_as<bool> decltype(auto) ret =
          std::ranges::starts_with(Iter1(a), Sent1(Iter1(a + 6)), Iter2(p), Sent2(Iter2(p + 6)));
      assert(ret);
    }
    {
      int a[]                               = {1, 2, 3, 4, 5, 6};
      int p[]                               = {1, 2, 3, 4, 5, 6};
      auto whole                            = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
      auto prefix                           = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 6)));
      std::same_as<bool> decltype(auto) ret = std::ranges::starts_with(whole, prefix);
      assert(ret);
    }
  }

  { // prefix is longer than range
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      int p[] = {1, 2, 3, 4, 5, 6, 7, 8};
      std::same_as<bool> decltype(auto) ret =
          std::ranges::starts_with(Iter1(a), Sent1(Iter1(a + 6)), Iter2(p), Sent2(Iter2(p + 8)));
      assert(!ret);
    }
    {
      int a[]                               = {1, 2, 3, 4, 5, 6};
      int p[]                               = {1, 2, 3, 4, 5, 6, 7, 8};
      auto whole                            = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
      auto prefix                           = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 8)));
      std::same_as<bool> decltype(auto) ret = std::ranges::starts_with(whole, prefix);
      assert(!ret);
    }
  }

  { // prefix has zero length
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      std::array<int, 0> p = {};
      std::same_as<bool> decltype(auto) ret =
          std::ranges::starts_with(Iter1(a), Sent1(Iter1(a + 6)), Iter2(p.data()), Sent2(Iter2(p.data())));
      assert(ret);
    }
    {
      int a[]                               = {1, 2, 3, 4, 5, 6};
      std::array<int, 0> p = {};
      auto whole                            = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
      auto prefix                           = std::ranges::subrange(Iter2(p.data()), Sent2(Iter2(p.data())));
      std::same_as<bool> decltype(auto) ret = std::ranges::starts_with(whole, prefix);
      assert(ret);
    }
  }

  { // range has zero length
    {
      std::array<int, 0> a = {};
      int p[] = {1, 2, 3, 4, 5, 6, 7, 8};
      std::same_as<bool> decltype(auto) ret =
          std::ranges::starts_with(Iter1(a.data()), Sent1(Iter1(a.data())), Iter2(p), Sent2(Iter2(p + 8)));
      assert(!ret);
    }
    {
      std::array<int, 0> a = {};
      int p[]                               = {1, 2, 3, 4, 5, 6, 7, 8};
      auto whole                            = std::ranges::subrange(Iter1(a.data()), Sent1(Iter1(a.data())));
      auto prefix                           = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 8)));
      std::same_as<bool> decltype(auto) ret = std::ranges::starts_with(whole, prefix);
      assert(!ret);
    }
  }

  { // check that the predicate is used
    {
      int a[] = {11, 8, 3, 4, 0, 6};
      int p[]                               = {1, 12};
      std::same_as<bool> decltype(auto) ret = std::ranges::starts_with(
          Iter1(a), Sent1(Iter1(a + 6)), Iter2(p), Sent2(Iter2(p + 2)), [](int l, int r) { return l > r; });
      assert(!ret);
    }
    {
      int a[]                               = {11, 8, 3, 4, 0, 6};
      int p[]                               = {1, 12};
      auto whole                            = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
      auto prefix                           = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 2)));
      std::same_as<bool> decltype(auto) ret = std::ranges::starts_with(whole, prefix, [](int l, int r) { return l > r; });
      assert(!ret);
    }
  }

  { // check that the projections are used
    {
      int a[]                               = {1, 3, 5, 1, 5, 6};
      int p[]                               = {2, 3, 4};
      std::same_as<bool> decltype(auto) ret = std::ranges::starts_with(
          Iter1(a),
          Sent1(Iter1(a + 6)),
          Iter2(p),
          Sent2(Iter2(p + 3)),
          {},
          [](int i) { return i + 3; },
          [](int i) { return i * 2; });
      assert(ret);
    }
    {
      int a[]                               = {1, 3, 5, 1, 5, 6};
      int p[]                               = {2, 3, 4};
      auto whole                            = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
      auto prefix                           = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
      std::same_as<bool> decltype(auto) ret = std::ranges::starts_with(
          whole, prefix, {}, [](int i) { return i + 3; }, [](int i) { return i * 2; });
      assert(ret);
    }
  }
}

constexpr bool test() {
  types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class Iter2>() {
    types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class Iter1>() {
      if constexpr (std::forward_iterator<Iter1> && std::forward_iterator<Iter2>)
        test_iterators<Iter1, Iter1, Iter2, Iter2>();
      if constexpr (std::forward_iterator<Iter2>)
        test_iterators<Iter1, sized_sentinel<Iter1>, Iter2, Iter2>();
      if constexpr (std::forward_iterator<Iter1>)
        test_iterators<Iter1, Iter1, Iter2, sized_sentinel<Iter2>>();
      test_iterators<Iter1, sized_sentinel<Iter1>, Iter2, sized_sentinel<Iter2>>();
    });
  });

  { // check that std::invoke is used
    struct S {
      int i;

      constexpr S identity() { return *this; }

      constexpr bool compare(const S& s) { return i == s.i; }
    };
    {
      S a[]    = {{1}, {2}, {3}, {4}};
      S p[]    = {{1}, {2}};
      auto ret = std::ranges::starts_with(a, a + 4, p, p + 2, &S::compare, &S::identity, &S::identity);
      assert(ret);
    }
    {
      S a[]    = {{1}, {2}, {3}, {4}};
      S p[]    = {{1}, {2}};
      auto ret = std::ranges::starts_with(a, p, &S::compare, &S::identity, &S::identity);
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
