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

// template<forward_iterator I1, sentinel_for<I1> S1, forward_iterator I2,
//          sentinel_for<I2> S2, class Pred = ranges::equal_to,
//          class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_comparable<I1, I2, Pred, Proj1, Proj2>
//   constexpr subrange<I1>
//     ranges::search(I1 first1, S1 last1, I2 first2, S2 last2, Pred pred = {},
//                    Proj1 proj1 = {}, Proj2 proj2 = {});
// template<forward_range R1, forward_range R2, class Pred = ranges::equal_to,
//          class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_comparable<iterator_t<R1>, iterator_t<R2>, Pred, Proj1, Proj2>
//   constexpr borrowed_subrange_t<R1>
//     ranges::search(R1&& r1, R2&& r2, Pred pred = {},
//                    Proj1 proj1 = {}, Proj2 proj2 = {});

#include <algorithm>
#include <array>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class Iter1, class Sent1 = Iter1, class Iter2 = int*, class Sent2 = Iter2>
concept HasSearchIt = requires (Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2) {
  std::ranges::search(first1, last1, first2, last2);
};

static_assert(HasSearchIt<int*>);
static_assert(!HasSearchIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasSearchIt<ForwardIteratorNotIncrementable>);
static_assert(!HasSearchIt<int*, SentinelForNotSemiregular>);
static_assert(!HasSearchIt<int*, int*, int**>); // not indirectly comparable
static_assert(!HasSearchIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasSearchIt<int*, int*, ForwardIteratorNotDerivedFrom>);
static_assert(!HasSearchIt<int*, int*, ForwardIteratorNotIncrementable>);
static_assert(!HasSearchIt<int*, int*, int*, SentinelForNotSemiregular>);
static_assert(!HasSearchIt<int*, int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Range1, class Range2 = UncheckedRange<int*>>
concept HasSearchR = requires (Range1 range1, Range2 range2) {
  std::ranges::search(range1, range2);
};

static_assert(HasSearchR<UncheckedRange<int*>>);
static_assert(!HasSearchR<ForwardRangeNotDerivedFrom>);
static_assert(!HasSearchR<ForwardIteratorNotIncrementable>);
static_assert(!HasSearchR<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasSearchR<ForwardRangeNotSentinelEqualityComparableWith>);
static_assert(!HasSearchR<UncheckedRange<int*>, UncheckedRange<int**>>); // not indirectly comparable
static_assert(!HasSearchR<UncheckedRange<int*>, ForwardRangeNotDerivedFrom>);
static_assert(!HasSearchR<UncheckedRange<int*>, ForwardRangeNotIncrementable>);
static_assert(!HasSearchR<UncheckedRange<int*>, ForwardRangeNotSentinelSemiregular>);
static_assert(!HasSearchR<UncheckedRange<int*>, ForwardRangeNotSentinelEqualityComparableWith>);

template <class Iter1, class Sent1, class Iter2, class Sent2 = Iter2>
constexpr void test_iterators() {
  { // simple test
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      int p[] = {3, 4};
      std::same_as<std::ranges::subrange<Iter1, Iter1>> decltype(auto) ret =
          std::ranges::search(Iter1(a), Sent1(Iter1(a + 6)), Iter2(p), Sent2(Iter2(p + 2)));
      assert(base(ret.begin()) == a + 2);
      assert(base(ret.end()) == a + 4);
    }
    {
      int a[] = {1, 2, 3, 4, 5, 6};
      int p[] = {3, 4};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
      auto range2 = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 2)));
      std::same_as<std::ranges::subrange<Iter1, Iter1>> decltype(auto) ret = std::ranges::search(range1, range2);
      assert(base(ret.begin()) == a + 2);
      assert(base(ret.end()) == a + 4);
    }
  }

  { // matching part begins at the front
    {
      int a[] = {7, 5, 3, 7, 3, 6};
      int p[] = {7, 5, 3};
      auto ret = std::ranges::search(Iter1(a), Sent1(Iter1(a + 6)), Iter2(p), Sent2(Iter2(p + 3)));
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 3);
    }
    {
      int a[] = {7, 5, 3, 7, 3, 6};
      int p[] = {7, 5, 3};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
      auto range2 = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
      auto ret = std::ranges::search(range1, range2);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 3);
    }
  }

  { // matching part ends at the back
    {
      int a[] = {9, 3, 6, 4, 8};
      int p[] = {4, 8};
      auto ret = std::ranges::search(Iter1(a), Sent1(Iter1(a + 5)), Iter2(p), Sent2(Iter2(p + 2)));
      assert(base(ret.begin()) == a + 3);
      assert(base(ret.end()) == a + 5);
    }
    {
      int a[] = {9, 3, 6, 4, 8};
      int p[] = {4, 8};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 5)));
      auto range2 = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 2)));
      auto ret = std::ranges::search(range1, range2);
      assert(base(ret.begin()) == a + 3);
      assert(base(ret.end()) == a + 5);
    }
  }

  { // pattern does not match
    {
      int a[] = {9, 3, 6, 4, 8};
      int p[] = {1};
      auto ret = std::ranges::search(Iter1(a), Sent1(Iter1(a + 5)), Iter2(p), Sent2(Iter2(p + 1)));
      assert(base(ret.begin()) == a + 5);
      assert(base(ret.end()) == a + 5);
    }
    {
      int a[] = {9, 3, 6, 4, 8};
      int p[] = {1};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 5)));
      auto range2 = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 1)));
      auto ret = std::ranges::search(range1, range2);
      assert(base(ret.begin()) == a + 5);
      assert(base(ret.end()) == a + 5);
    }
  }

  { // range and pattern are identical
    {
      int a[] = {6, 7, 8, 9};
      int p[] = {6, 7, 8, 9};
      auto ret = std::ranges::search(Iter1(a), Sent1(Iter1(a + 4)), Iter2(p), Sent2(Iter2(p + 4)));
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 4);
    }
    {
      int a[] = {6, 7, 8, 9};
      int p[] = {6, 7, 8, 9};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 4)));
      auto range2 = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 4)));
      auto ret = std::ranges::search(range1, range2);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 4);
    }
  }

  { // pattern is longer than range
    {
      int a[] = {6, 7, 8};
      int p[] = {6, 7, 8, 9};
      auto ret = std::ranges::search(Iter1(a), Sent1(Iter1(a + 3)), Iter2(p), Sent2(Iter2(p + 4)));
      assert(base(ret.begin()) == a + 3);
      assert(base(ret.end()) == a + 3);
    }
    {
      int a[] = {6, 7, 8};
      int p[] = {6, 7, 8, 9};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 3)));
      auto range2 = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 4)));
      auto ret = std::ranges::search(range1, range2);
      assert(base(ret.begin()) == a + 3);
      assert(base(ret.end()) == a + 3);
    }
  }

  { // pattern has zero length
    {
      int a[] = {6, 7, 8};
      int p[] = {};
      auto ret = std::ranges::search(Iter1(a), Sent1(Iter1(a + 3)), Iter2(p), Sent2(Iter2(p)));
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a);
    }
    {
      int a[] = {6, 7, 8};
      int p[] = {};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 3)));
      auto range2 = std::ranges::subrange(Iter2(p), Sent2(Iter2(p)));
      auto ret = std::ranges::search(range1, range2);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a);
    }
  }

  { // range has zero length
    {
      int a[] = {};
      int p[] = {6, 7, 8};
      auto ret = std::ranges::search(Iter1(a), Sent1(Iter1(a)), Iter2(p), Sent2(Iter2(p + 3)));
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a);
    }
    {
      int a[] = {};
      int p[] = {6, 7, 8};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a)));
      auto range2 = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
      auto ret = std::ranges::search(range1, range2);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a);
    }
  }

  { // check that the first match is returned
    {
      int a[] = {6, 7, 8, 6, 7, 8, 6, 7, 8};
      int p[] = {6, 7, 8};
      auto ret = std::ranges::search(Iter1(a), Sent1(Iter1(a + 9)), Iter2(p), Sent2(Iter2(p + 3)));
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 3);
    }
    {
      int a[] = {6, 7, 8, 6, 7, 8, 6, 7, 8};
      int p[] = {6, 7, 8};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 9)));
      auto range2 = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
      auto ret = std::ranges::search(range1, range2);
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 3);
    }
  }

  { // check that the predicate is used
    {
      int a[] = {1, 2, 8, 1, 5, 6};
      int p[] = {7, 0, 4};
      auto ret = std::ranges::search(Iter1(a), Sent1(Iter1(a + 6)),
                                     Iter2(p), Sent2(Iter2(p + 3)),
                                     [](int l, int r) { return l > r; });
      assert(base(ret.begin()) == a + 2);
      assert(base(ret.end()) == a + 5);
    }
    {
      int a[] = {1, 2, 8, 1, 5, 6};
      int p[] = {7, 0, 4};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
      auto range2 = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
      auto ret = std::ranges::search(range1, range2, [](int l, int r) { return l > r; });
      assert(base(ret.begin()) == a + 2);
      assert(base(ret.end()) == a + 5);
    }
  }

  { // check that the projections are used
    {
      int a[] = {1, 3, 5, 1, 5, 6};
      int p[] = {2, 3, 4};
      auto ret = std::ranges::search(Iter1(a), Sent1(Iter1(a + 6)),
                                     Iter2(p), Sent2(Iter2(p + 3)),
                                     {},
                                     [](int i) { return i + 3; },
                                     [](int i) { return i * 2; });
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 3);
    }
    {
      int a[] = {1, 3, 5, 1, 5, 6};
      int p[] = {2, 3, 4};
      auto range1 = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
      auto range2 = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
      auto ret = std::ranges::search(range1,
                                     range2,
                                     {},
                                     [](int i) { return i + 3; },
                                     [](int i) { return i * 2; });
      assert(base(ret.begin()) == a);
      assert(base(ret.end()) == a + 3);
    }
  }
}

template <class Iter1, class Sent1 = Iter1>
constexpr void test_iterators2() {
  test_iterators<Iter1, Sent1, forward_iterator<int*>>();
  test_iterators<Iter1, Sent1, forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>();
  test_iterators<Iter1, Sent1, bidirectional_iterator<int*>>();
  test_iterators<Iter1, Sent1, bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>();
  test_iterators<Iter1, Sent1, random_access_iterator<int*>>();
  test_iterators<Iter1, Sent1, random_access_iterator<int*>, sized_sentinel<random_access_iterator<int*>>>();
  test_iterators<Iter1, Sent1, contiguous_iterator<int*>>();
  test_iterators<Iter1, Sent1, contiguous_iterator<int*>, sized_sentinel<contiguous_iterator<int*>>>();
  test_iterators<Iter1, Sent1, int*>();
}

constexpr bool test() {
  test_iterators2<forward_iterator<int*>>();
  test_iterators2<forward_iterator<int*>, sized_sentinel<forward_iterator<int*>>>();
  test_iterators2<bidirectional_iterator<int*>>();
  test_iterators2<bidirectional_iterator<int*>, sized_sentinel<bidirectional_iterator<int*>>>();
  test_iterators2<random_access_iterator<int*>>();
  test_iterators2<random_access_iterator<int*>, sized_sentinel<random_access_iterator<int*>>>();
  test_iterators2<contiguous_iterator<int*>>();
  test_iterators2<contiguous_iterator<int*>, sized_sentinel<contiguous_iterator<int*>>>();
  test_iterators2<int*>();

  { // check that std::invoke is used
    struct S {
      int i;

      constexpr S identity() {
        return *this;
      }

      constexpr bool compare(const S& s) {
        return i == s.i;
      }
    };
    {
      S a[] = {{1}, {2}, {3}, {4}};
      S p[] = {{2}, {3}};
      auto ret = std::ranges::search(a, a + 4, p, p + 2, &S::compare, &S::identity, &S::identity);
      assert(ret.begin() == a + 1);
      assert(ret.end() == a + 3);
    }
    {
      S a[] = {{1}, {2}, {3}, {4}};
      S p[] = {{2}, {3}};
      auto ret = std::ranges::search(a, p, &S::compare, &S::identity, &S::identity);
      assert(ret.begin() == a + 1);
      assert(ret.end() == a + 3);
    }
  }

  { // check that std::ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) ret =
        std::ranges::search(std::array {1, 2, 3, 4}, std::array {2, 3});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
