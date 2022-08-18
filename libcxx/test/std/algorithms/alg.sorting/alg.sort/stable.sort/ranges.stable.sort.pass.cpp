//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<random_access_iterator I, sentinel_for<I> S, class Comp = ranges::less,
//         class Proj = identity>
//   requires sortable<I, Comp, Proj>
//   I ranges::stable_sort(I first, S last, Comp comp = {}, Proj proj = {});                 // since C++20
//
// template<random_access_range R, class Comp = ranges::less, class Proj = identity>
//   requires sortable<iterator_t<R>, Comp, Proj>
//   borrowed_iterator_t<R>
//     ranges::stable_sort(R&& r, Comp comp = {}, Proj proj = {});                           // since C++20

#include <algorithm>
#include <array>
#include <concepts>
#include <functional>
#include <ranges>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

// SFINAE tests.

using BadComparator = ComparatorNotCopyable<int*>;
static_assert(!std::sortable<int*, BadComparator>);

template <class Iter, class Sent = sentinel_wrapper<Iter>, class Comp = std::ranges::less>
concept HasStableSortIt = requires(Iter first, Sent last, Comp comp) { std::ranges::stable_sort(first, last, comp); };

static_assert(HasStableSortIt<int*>);
static_assert(!HasStableSortIt<RandomAccessIteratorNotDerivedFrom>);
static_assert(!HasStableSortIt<RandomAccessIteratorBadIndex>);
static_assert(!HasStableSortIt<int*, SentinelForNotSemiregular>);
static_assert(!HasStableSortIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasStableSortIt<int*, int*, BadComparator>);
static_assert(!HasStableSortIt<const int*>); // Doesn't satisfy `sortable`.

template <class Range, class Comp = std::ranges::less>
concept HasStableSortR = requires(Range range, Comp comp) { std::ranges::stable_sort(range, comp); };

static_assert(HasStableSortR<UncheckedRange<int*>>);
static_assert(!HasStableSortR<RandomAccessRangeNotDerivedFrom>);
static_assert(!HasStableSortR<RandomAccessRangeBadIndex>);
static_assert(!HasStableSortR<UncheckedRange<int*, SentinelForNotSemiregular>>);
static_assert(!HasStableSortR<UncheckedRange<int*, SentinelForNotWeaklyEqualityComparableWith>>);
static_assert(!HasStableSortR<UncheckedRange<int*>, BadComparator>);
static_assert(!HasStableSortR<UncheckedRange<const int*>>); // Doesn't satisfy `sortable`.

template <class Iter, class Sent, size_t N>
void test_one(std::array<int, N> input, std::array<int, N> expected) {
  { // (iterator, sentinel) overload.
    auto sorted = input;
    auto b = Iter(sorted.data());
    auto e = Sent(Iter(sorted.data() + sorted.size()));

    std::same_as<Iter> decltype(auto) last = std::ranges::stable_sort(b, e);
    assert(sorted == expected);
    assert(base(last) == sorted.data() + sorted.size());
  }

  { // (range) overload.
    auto sorted = input;
    auto b = Iter(sorted.data());
    auto e = Sent(Iter(sorted.data() + sorted.size()));
    auto range = std::ranges::subrange(b, e);

    std::same_as<Iter> decltype(auto) last = std::ranges::stable_sort(range);
    assert(sorted == expected);
    assert(base(last) == sorted.data() + sorted.size());
  }
}

template <class Iter, class Sent>
void test_iterators_2() {
  // Empty sequence.
  test_one<Iter, Sent, 0>({}, {});
  // 1-element sequence.
  test_one<Iter, Sent, 1>({1}, {1});
  // 2-element sequence.
  test_one<Iter, Sent, 2>({2, 1}, {1, 2});
  // 3-element sequence.
  test_one<Iter, Sent, 3>({2, 1, 3}, {1, 2, 3});
  // Longer sequence.
  test_one<Iter, Sent, 8>({2, 1, 3, 6, 8, 4, 11, 5}, {1, 2, 3, 4, 5, 6, 8, 11});
  // Longer sequence with duplicates.
  test_one<Iter, Sent, 7>({2, 1, 3, 6, 2, 8, 6}, {1, 2, 2, 3, 6, 6, 8});
  // All elements are the same.
  test_one<Iter, Sent, 3>({1, 1, 1}, {1, 1, 1});
  // Already sorted.
  test_one<Iter, Sent, 5>({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5});
  // Reverse-sorted.
  test_one<Iter, Sent, 5>({5, 4, 3, 2, 1}, {1, 2, 3, 4, 5});
  // Repeating pattern.
  test_one<Iter, Sent, 6>({1, 2, 1, 2, 1, 2}, {1, 1, 1, 2, 2, 2});
}

template <class Iter>
void test_iterators_1() {
  test_iterators_2<Iter, Iter>();
  test_iterators_2<Iter, sentinel_wrapper<Iter>>();
}

void test_iterators() {
  test_iterators_1<random_access_iterator<int*>>();
  test_iterators_1<contiguous_iterator<int*>>();
  test_iterators_1<int*>();
}

void test() {
  test_iterators();

  struct OrderedValue {
    int value;
    double original_order;
    bool operator==(const OrderedValue&) const = default;
    auto operator<=>(const OrderedValue& rhs) const { return value <=> rhs.value; }
  };

  { // The sort is stable (equivalent elements remain in the same order).
    using V = OrderedValue;
    using Array = std::array<V, 20>;
    Array orig_in = {
      V{10, 10.1}, {12, 12.1}, {3, 3.1}, {5, 5.1}, {3, 3.2}, {3, 3.3}, {11, 11.1}, {12, 12.2}, {4, 4.1}, {4, 4.2},
      {4, 4.3}, {1, 1.1}, {6, 6.1}, {3, 3.4}, {10, 10.2}, {8, 8.1}, {12, 12.3}, {1, 1.2}, {1, 1.3}, {5, 5.2}
    };
    Array expected = {
      V{1, 1.1}, {1, 1.2}, {1, 1.3},
      {3, 3.1}, {3, 3.2}, {3, 3.3}, {3, 3.4},
      {4, 4.1}, {4, 4.2}, {4, 4.3},
      {5, 5.1}, {5, 5.2},
      {6, 6.1},
      {8, 8.1},
      {10, 10.1}, {10, 10.2},
      {11, 11.1},
      {12, 12.1}, {12, 12.2}, {12, 12.3}
    };

    {
      auto in = orig_in;
      auto last = std::ranges::stable_sort(in.begin(), in.end());
      assert(in == expected);
      assert(last == in.end());
    }

    {
      auto in = orig_in;
      auto last = std::ranges::stable_sort(in);
      assert(in == expected);
      assert(last == in.end());
    }
  }

  { // A custom comparator works and is stable.
    using V = OrderedValue;
    using Array = std::array<V, 11>;

    Array orig_in = {
      V{1, 1.1}, {2, 2.1}, {2, 2.2}, {3, 3.1}, {2, 2.3}, {3, 3.2}, {4, 4.1}, {5, 5.1}, {2, 2.4}, {5, 5.2}, {1, 1.2}
    };
    Array expected = {
      V{5, 5.1}, {5, 5.2},
      {4, 4.1},
      {3, 3.1}, {3, 3.2},
      {2, 2.1}, {2, 2.2}, {2, 2.3}, {2, 2.4},
      {1, 1.1}, {1, 1.2}
    };

    {
      auto in = orig_in;
      auto last = std::ranges::stable_sort(in.begin(), in.end(), std::greater{});
      assert(in == expected);
      assert(last == in.end());
    }

    {
      auto in = orig_in;
      auto last = std::ranges::stable_sort(in, std::greater{});
      assert(in == expected);
      assert(last == in.end());
    }
  }

  { // A custom projection works.
    struct A {
      int a;
      bool operator==(const A&) const = default;
    };

    {
      std::array in = {A{2}, A{3}, A{1}};
      auto last = std::ranges::stable_sort(in.begin(), in.end(), {}, &A::a);
      assert((in == std::array{A{1}, A{2}, A{3}}));
      assert(last == in.end());
    }

    {
      std::array in = {A{2}, A{3}, A{1}};
      auto last = std::ranges::stable_sort(in, {}, &A::a);
      assert((in == std::array{A{1}, A{2}, A{3}}));
      assert(last == in.end());
    }
  }

  { // `std::invoke` is used in the implementation.
    struct S {
      int i;
      S(int i_) : i(i_) {}

      bool comparator(const S& rhs) const { return i < rhs.i; }
      const S& projection() const { return *this; }

      bool operator==(const S&) const = default;
    };

    {
      std::array in = {S{2}, S{3}, S{1}};
      auto last = std::ranges::stable_sort(in.begin(), in.end(), &S::comparator, &S::projection);
      assert((in == std::array{S{1}, S{2}, S{3}}));
      assert(last == in.end());
    }

    {
      std::array in = {S{2}, S{3}, S{1}};
      auto last = std::ranges::stable_sort(in, &S::comparator, &S::projection);
      assert((in == std::array{S{1}, S{2}, S{3}}));
      assert(last == in.end());
    }
  }

  { // The comparator can return any type that's convertible to `bool`.
    {
      std::array in = {2, 1, 3};
      auto last = std::ranges::stable_sort(in.begin(), in.end(), [](int i, int j) { return BooleanTestable{i < j}; });
      assert((in == std::array{1, 2, 3}));
      assert(last == in.end());
    }

    {
      std::array in = {2, 1, 3};
      auto last = std::ranges::stable_sort(in, [](int i, int j) { return BooleanTestable{i < j}; });
      assert((in == std::array{1, 2, 3}));
      assert(last == in.end());
    }
  }

  { // `std::ranges::dangling` is returned.
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) result =
        std::ranges::stable_sort(std::array{1, 2, 3});
  }

  // TODO: Enable the tests once the implementation switched to use iter_move/iter_swap
  /*
  { // ProxyIterator
    {
      std::array in = {2, 1, 3};
      ProxyRange proxy{in};

      std::ranges::stable_sort(proxy.begin(), proxy.end(), [](auto i, auto j) { return i.data < j.data; });
      assert((in == std::array{1, 2, 3}));
    }

    {
      std::array in = {2, 1, 3};
      ProxyRange proxy{in};
      std::ranges::stable_sort(proxy, [](auto i, auto j) { return i.data < j.data; });
      assert((in == std::array{1, 2, 3}));
    }
  }
  */
}

int main(int, char**) {
  test();
  // Note: `stable_sort` is not `constexpr`.

  return 0;
}
