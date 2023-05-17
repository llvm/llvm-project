//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_iterator I, sentinel_for<I> S, class T1, class T2, class Proj = identity>
//   requires indirectly_writable<I, const T2&> &&
//            indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T1*>
//   constexpr I
//     ranges::replace(I first, S last, const T1& old_value, const T2& new_value, Proj proj = {});
// template<input_range R, class T1, class T2, class Proj = identity>
//   requires indirectly_writable<iterator_t<R>, const T2&> &&
//            indirect_binary_predicate<ranges::equal_to, projected<iterator_t<R>, Proj>, const T1*>
//   constexpr borrowed_iterator_t<R>
//     ranges::replace(R&& r, const T1& old_value, const T2& new_value, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

template <class Iter, class Sent = sentinel_wrapper<Iter>>
concept HasReplaceIt = requires(Iter iter, Sent sent) { std::ranges::replace(iter, sent, 0, 0); };

static_assert(HasReplaceIt<int*>);
static_assert(!HasReplaceIt<InputIteratorNotDerivedFrom>);
static_assert(!HasReplaceIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasReplaceIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasReplaceIt<int*, SentinelForNotSemiregular>);
static_assert(!HasReplaceIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasReplaceIt<int**>); // not indirectly_writable
static_assert(!HasReplaceIt<IndirectBinaryPredicateNotIndirectlyReadable>);

template <class Range>
concept HasReplaceR = requires(Range range) { std::ranges::replace(range, 0, 0); };

static_assert(HasReplaceR<UncheckedRange<int*>>);
static_assert(!HasReplaceR<InputRangeNotDerivedFrom>);
static_assert(!HasReplaceR<InputRangeNotIndirectlyReadable>);
static_assert(!HasReplaceR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasReplaceR<InputRangeNotSentinelSemiregular>);
static_assert(!HasReplaceR<InputRangeNotSentinelEqualityComparableWith>);
static_assert(!HasReplaceR<UncheckedRange<int**>>); // not indirectly_writable
static_assert(!HasReplaceR<InputRangeIndirectBinaryPredicateNotIndirectlyReadable>);

template <int N>
struct Data {
  std::array<int, N> input;
  int oldval;
  int newval;
  std::array<int, N> expected;
};

template <class Iter, class Sent, int N>
constexpr void test(Data<N> d) {
  {
    auto a = d.input;
    std::same_as<Iter> decltype(auto) ret = std::ranges::replace(Iter(a.data()), Sent(Iter(a.data() + N)),
                                                                 d.oldval,
                                                                 d.newval);
    assert(base(ret) == a.data() + N);
    assert(a == d.expected);
  }
  {
    auto a = d.input;
    auto range = std::ranges::subrange(Iter(a.data()), Sent(Iter(a.data() + N)));
    std::same_as<Iter> decltype(auto) ret = std::ranges::replace(range, d.oldval, d.newval);
    assert(base(ret) == a.data() + N);
    assert(a == d.expected);
  }
}

template <class Iter, class Sent = Iter>
constexpr void test_iterators() {
  // simple test
  test<Iter, Sent, 4>({.input = {1, 2, 3, 4}, .oldval = 2, .newval = 23, .expected = {1, 23, 3, 4}});
  // no match
  test<Iter, Sent, 4>({.input = {1, 2, 3, 4}, .oldval = 5, .newval = 23, .expected = {1, 2, 3, 4}});
  // all match
  test<Iter, Sent, 4>({.input = {1, 1, 1, 1}, .oldval = 1, .newval = 23, .expected = {23, 23, 23, 23}});
  // empty range
  test<Iter, Sent, 0>({.input = {}, .oldval = 1, .newval = 23, .expected = {}});
  // single element range
  test<Iter, Sent, 1>({.input = {1}, .oldval = 1, .newval = 2, .expected = {2}});
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
      std::ranges::replace(a, a + 4, 3, S{0}, &S::i);
    }
    {
      S a[] = {1, 2, 3, 4};
      std::ranges::replace(a, 3, S{0}, &S::i);
    }
  }

  { // check that std::invoke is used
    struct S {
      constexpr S(int i_) : i(i_) {}
      constexpr bool operator==(const S&) const = default;
      constexpr const S& identity() const { return *this; }
      int i;
    };
    {
      S a[] = {1, 2, 3, 4};
      auto ret = std::ranges::replace(std::begin(a), std::end(a), S{1}, S{2}, &S::identity);
      assert(ret == a + 4);
    }
    {
      S a[] = {1, 2, 3, 4};
      auto ret = std::ranges::replace(a, S{1}, S{2}, &S::identity);
      assert(ret == a + 4);
    }
  }

  { // check that the implicit conversion to bool works
    {
      StrictComparable<int> a[] = {1, 2, 2, 4};
      auto ret = std::ranges::replace(std::begin(a), std::end(a), 1, 2);
      assert(ret == std::end(a));
    }
    {
      StrictComparable<int> a[] = {1, 2, 2, 4};
      auto ret = std::ranges::replace(a, 1, 2);
      assert(ret == std::end(a));
    }
  }

  { // check that T1 and T2 can be different types
    {
      StrictComparable<int> a[] = {1, 2, 2, 4};
      auto ret = std::ranges::replace(std::begin(a), std::end(a), '\0', 2ull);
      assert(ret == std::end(a));
    }
    {
      StrictComparable<int> a[] = {1, 2, 2, 4};
      auto ret = std::ranges::replace(a, '\0', 2ull);
      assert(ret == std::end(a));
    }
  }

  { // check that std::ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) ret =
        std::ranges::replace(std::array {1, 2, 3, 4}, 1, 1);
  }

  { // check that the complexity requirements are met
    {
      auto projectionCount = 0;
      auto proj = [&](int i) { ++projectionCount; return i; };
      int a[] = {1, 2, 3, 4, 5};
      auto ret = std::ranges::replace(std::begin(a), std::end(a), 1, 2, proj);
      assert(ret == a + 5);
      assert(projectionCount == 5);
    }
    {
      auto projectionCount = 0;
      auto proj = [&](int i) { ++projectionCount; return i; };
      int a[] = {1, 2, 3, 4, 5};
      auto ret = std::ranges::replace(a, 1, 2, proj);
      assert(ret == a + 5);
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
