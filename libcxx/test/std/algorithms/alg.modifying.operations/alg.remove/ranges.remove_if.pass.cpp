//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<permutable I, sentinel_for<I> S, class Proj = identity,
//          indirect_unary_predicate<projected<I, Proj>> Pred>
//   constexpr subrange<I> ranges::remove_if(I first, S last, Pred pred, Proj proj = {});
// template<forward_range R, class Proj = identity,
//          indirect_unary_predicate<projected<iterator_t<R>, Proj>> Pred>
//   requires permutable<iterator_t<R>>
//   constexpr borrowed_subrange_t<R>
//     ranges::remove_if(R&& r, Pred pred, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct FalsePredicate {
  bool operator()(int) { return false; }
};

template <class Iter, class Sent = sentinel_wrapper<Iter>>
concept HasRemoveIfIt = requires(Iter first, Sent last) { std::ranges::remove_if(first, last, FalsePredicate{}); };

static_assert(HasRemoveIfIt<int*>);
static_assert(!HasRemoveIfIt<PermutableNotForwardIterator>);
static_assert(!HasRemoveIfIt<PermutableNotSwappable>);
static_assert(!HasRemoveIfIt<int*, SentinelForNotSemiregular>);
static_assert(!HasRemoveIfIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasRemoveIfIt<int**>); // not indirect_unary_predicate

template <class Range>
concept HasRemoveIfR = requires(Range range) { std::ranges::remove_if(range, FalsePredicate{}); };

static_assert(HasRemoveIfR<UncheckedRange<int*>>);
static_assert(!HasRemoveIfR<PermutableRangeNotForwardIterator>);
static_assert(!HasRemoveIfR<PermutableRangeNotSwappable>);
static_assert(!HasRemoveIfR<SentinelForNotSemiregular>);
static_assert(!HasRemoveIfR<SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasRemoveIfR<UncheckedRange<int**>>); // not indirect_unary_predicate

template <int N, int M>
struct Data {
  std::array<int, N> input;
  std::array<int, M> expected;
  int cutoff;
};

template <class Iter, class Sent, int N, int M>
constexpr void test(Data<N, M> d) {
  { // iterator overload
    auto input = d.input;

    auto first = Iter(input.data());
    auto last  = Sent(Iter(input.data() + input.size()));

    std::same_as<std::ranges::subrange<Iter>> decltype(auto) ret =
        std::ranges::remove_if(std::move(first), std::move(last), [&](int i) { return i < d.cutoff; });

    assert(base(ret.begin()) == input.data() + M);
    assert(base(ret.end()) == input.data() + N);
    assert(std::ranges::equal(input.data(), base(ret.begin()), d.expected.begin(), d.expected.end()));
  }

  { // range overload
    auto input = d.input;
    auto range = std::ranges::subrange(Iter(input.data()), Sent(Iter(input.data() + input.size())));

    std::same_as<std::ranges::subrange<Iter>> decltype(auto) ret = std::ranges::remove_if(range, [&](int i) {
      return i < d.cutoff;
    });

    assert(base(ret.begin()) == input.data() + M);
    assert(base(ret.end()) == input.data() + N);
    assert(std::ranges::equal(input.data(), base(ret.begin()), d.expected.begin(), d.expected.end()));
  }
}

template <class Iter, class Sent>
constexpr void tests() {
  // simple test
  test<Iter, Sent, 6, 2>({.input = {1, 2, 3, 4, 5, 6}, .expected = {5, 6}, .cutoff = 5});
  // empty range
  test<Iter, Sent, 0, 0>({});
  // single element range - no match
  test<Iter, Sent, 1, 1>({.input = {1}, .expected = {1}, .cutoff = 1});
  // single element range - match
  test<Iter, Sent, 1, 0>({.input = {1}, .expected = {}, .cutoff = 2});
  // two element range
  test<Iter, Sent, 2, 1>({.input = {1, 2}, .expected = {2}, .cutoff = 2});
  // all elements match
  test<Iter, Sent, 5, 0>({.input = {1, 1, 1, 1, 1}, .expected = {}, .cutoff = 2});
  // no elements match
  test<Iter, Sent, 5, 5>({.input = {1, 1, 1, 1, 1}, .expected = {1, 1, 1, 1, 1}, .cutoff = 0});
  // the relative order of elements isn't changed
  test<Iter, Sent, 8, 7>({.input = {1, 2, 3, 2, 3, 4, 2, 5}, .expected = {2, 3, 2, 3, 4, 2, 5}, .cutoff = 2});
  // multiple matches in a row
  test<Iter, Sent, 5, 3>({.input = {1, 2, 2, 2, 1}, .expected = {2, 2, 2}, .cutoff = 2});
  // only the last element matches
  test<Iter, Sent, 3, 2>({.input = {2, 2, 1}, .expected = {2, 2}, .cutoff = 2});
  // only the last element doesn't match
  test<Iter, Sent, 3, 1>({.input = {1, 1, 2}, .expected = {2}, .cutoff = 2});
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
        std::ranges::remove_if(std::array{1, 2, 3, 4}, [](int i) { return i < 0; });
  }

  {// check complexity requirements

   // This is https://llvm.org/PR56382 - clang-format behaves weird if function-local structs are used
   // clang-format off
    {
      int comp_count = 0;
      auto comp = [&](int i) {
        ++comp_count;
        return i == 0;
      };
      int proj_count = 0;
      auto proj      = [&](int i) {
        ++proj_count;
        return i;
      };
      int a[]  = {1, 2, 3, 4};
      auto ret = std::ranges::remove_if(std::begin(a), std::end(a), comp, proj);
      assert(ret.begin() == std::end(a) && ret.end() == std::end(a));
      assert(comp_count == 4);
      assert(proj_count == 4);
    }
    {
      int comp_count = 0;
      auto comp      = [&](int i) {
        ++comp_count;
        return i == 0;
      };
      int proj_count = 0;
      auto proj      = [&](int i) {
        ++proj_count;
        return i;
      };
      int a[]  = {1, 2, 3, 4};
      auto ret = std::ranges::remove_if(a, comp, proj);
      assert(ret.begin() == std::end(a) && ret.end() == std::end(a));
      assert(comp_count == 4);
      assert(proj_count == 4);
    }
  }

  { // check that std::invoke is used
    struct S {
      constexpr S& identity() { return *this; }
      constexpr bool predicate() const { return true; }
    };
    {
      S a[4]   = {};
      auto ret = std::ranges::remove_if(std::begin(a), std::end(a), &S::predicate, &S::identity);
      assert(ret.begin() == std::begin(a));
      assert(ret.end() == std::end(a));
    }
    {
      S a[4]   = {};
      auto ret = std::ranges::remove_if(a, &S::predicate, &S::identity);
      assert(ret.begin() == std::begin(a));
      assert(ret.end() == std::end(a));
    }
  }

  return true;
}
// clang-format on

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
