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

// template<permutable I, sentinel_for<I> S, class T, class Proj = identity>
//   requires indirect_binary_predicate<ranges::equal_to, projected<I, Proj>, const T*>
//   constexpr subrange<I> ranges::remove(I first, S last, const T& value, Proj proj = {});
// template<forward_range R, class T, class Proj = identity>
//   requires permutable<iterator_t<R>> &&
//            indirect_binary_predicate<ranges::equal_to, projected<iterator_t<R>, Proj>, const T*>
//   constexpr borrowed_subrange_t<R>
//     ranges::remove(R&& r, const T& value, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "boolean_testable.h"
#include "test_iterators.h"

template <class Iter, class Sent = sentinel_wrapper<Iter>>
concept HasRemoveIt = requires(Iter first, Sent last) { std::ranges::remove(first, last, 0); };

static_assert(HasRemoveIt<int*>);
static_assert(!HasRemoveIt<PermutableNotForwardIterator>);
static_assert(!HasRemoveIt<PermutableNotSwappable>);
static_assert(!HasRemoveIt<int*, SentinelForNotSemiregular>);
static_assert(!HasRemoveIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasRemoveIt<int**>); // not indirect_binary_prediacte

template <class Range>
concept HasRemoveR = requires(Range range) { std::ranges::remove(range, 0); };

static_assert(HasRemoveR<UncheckedRange<int*>>);
static_assert(!HasRemoveR<PermutableRangeNotForwardIterator>);
static_assert(!HasRemoveR<PermutableRangeNotSwappable>);
static_assert(!HasRemoveR<SentinelForNotSemiregular>);
static_assert(!HasRemoveR<SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasRemoveR<UncheckedRange<int**>>); // not indirect_binary_prediacte

template <int N, int M>
struct Data {
  std::array<int, N> input;
  std::array<int, M> expected;
  int val;
};

template <class Iter, class Sent, int N, int M>
constexpr void test(Data<N, M> d) {
  { // iterator overload
    auto input = d.input;

    std::same_as<std::ranges::subrange<Iter>> decltype(auto) ret =
        std::ranges::remove(Iter(input.data()), Sent(Iter(input.data() + input.size())), d.val);

    assert(base(ret.begin()) == input.data() + M);
    assert(base(ret.end()) == input.data() + N);
    assert(std::ranges::equal(input.begin(), base(ret.begin()), d.expected.begin(), d.expected.end()));
  }

  { // range overload
    auto input = d.input;
    auto range = std::ranges::subrange(Iter(input.data()), Sent(Iter(input.data() + input.size())));

    std::same_as<std::ranges::subrange<Iter>> decltype(auto) ret = std::ranges::remove(range, d.val);

    assert(base(ret.begin()) == input.data() + M);
    assert(base(ret.end()) == input.data() + N);
    assert(std::ranges::equal(base(input.begin()), base(ret.begin()), d.expected.begin(), d.expected.end()));
  }
}

template <class Iter, class Sent>
constexpr void tests() {
  // simple test
  test<Iter, Sent, 6, 5>({.input = {1, 2, 3, 4, 5, 6}, .expected = {1, 2, 3, 4, 6}, .val = 5});
  // empty range
  test<Iter, Sent, 0, 0>({});
  // single element range - match
  test<Iter, Sent, 1, 0>({.input = {1}, .expected = {}, .val = 1});
  // single element range - no match
  test<Iter, Sent, 1, 1>({.input = {1}, .expected = {1}, .val = 2});
  // two element range - same order
  test<Iter, Sent, 2, 1>({.input = {1, 2}, .expected = {1}, .val = 2});
  // two element range - reversed order
  test<Iter, Sent, 2, 1>({.input = {1, 2}, .expected = {2}, .val = 1});
  // all elements match
  test<Iter, Sent, 5, 0>({.input = {1, 1, 1, 1, 1}, .expected = {}, .val = 1});
  // the relative order of elements isn't changed
  test<Iter, Sent, 8, 5>({.input = {1, 2, 3, 2, 3, 4, 2, 5}, .expected = {1, 3, 3, 4, 5}, .val = 2});
}

template <class Iter>
constexpr void test_sentinels() {
  tests<Iter, Iter>();
  tests<Iter, sentinel_wrapper<Iter>>();
  tests<Iter, sized_sentinel<Iter>>();
}

constexpr void test_iterators() {
  test_sentinels<forward_iterator<int*>>();
  test_sentinels<bidirectional_iterator<int*>>();
  test_sentinels<random_access_iterator<int*>>();
  test_sentinels<contiguous_iterator<int*>>();
  test_sentinels<int*>();
}

constexpr bool test() {
  test_iterators();

  { // check that ranges::dangling is returned
    [[maybe_unused]] std::same_as<std::ranges::dangling> decltype(auto) ret =
        std::ranges::remove(std::array{1, 2, 3, 4}, 1);
  }

  { // check complexity requirements
    struct CompCounter {
      int* comp_count;

      constexpr bool operator==(const CompCounter&) const {
        ++*comp_count;
        return false;
      }
    };
    {
      int proj_count = 0;
      auto proj      = [&](CompCounter i) {
        ++proj_count;
        return i;
      };
      int comp_count = 0;

      CompCounter a[] = {{&comp_count}, {&comp_count}, {&comp_count}, {&comp_count}};
      auto ret        = std::ranges::remove(std::begin(a), std::end(a), CompCounter{&comp_count}, proj);
      assert(ret.begin() == std::end(a) && ret.end() == std::end(a));
      assert(comp_count == 4);
      assert(proj_count == 4);
    }
    {
      int proj_count = 0;
      auto proj      = [&](CompCounter i) {
        ++proj_count;
        return i;
      };
      int comp_count = 0;

      CompCounter a[] = {{&comp_count}, {&comp_count}, {&comp_count}, {&comp_count}};
      auto ret        = std::ranges::remove(a, CompCounter{&comp_count}, proj);
      assert(ret.begin() == std::end(a) && ret.end() == std::end(a));
      assert(comp_count == 4);
      assert(proj_count == 4);
    }
  }

  { // check that std::invoke is used
    struct S {
      constexpr S& identity() { return *this; }
      bool operator==(const S&) const = default;
    };
    {
      S a[4]   = {};
      auto ret = std::ranges::remove(std::begin(a), std::end(a), S{}, &S::identity);
      assert(ret.begin() == std::begin(a));
      assert(ret.end() == std::end(a));
    }
    {
      S a[4]   = {};
      auto ret = std::ranges::remove(a, S{}, &S::identity);
      assert(ret.begin() == std::begin(a));
      assert(ret.end() == std::end(a));
    }
  }

  {
    // check that the implicit conversion to bool works
    {
      StrictComparable<int> a[] = {1, 2, 3, 4};
      auto ret                  = std::ranges::remove(a, a + 4, StrictComparable<int>{2});
      assert(ret.begin() == a + 3);
    }
    {
      StrictComparable<int> a[] = {1, 2, 3, 4};
      auto ret                  = std::ranges::remove(a, StrictComparable<int>{2});
      assert(ret.begin() == a + 3);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
