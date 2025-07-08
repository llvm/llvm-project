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
//          indirectly_unary_invocable<projected<I, Proj>> Fun>
//   constexpr ranges::for_each_result<I, Fun>
//     ranges::for_each(I first, S last, Fun f, Proj proj = {});
// template<input_range R, class Proj = identity,
//          indirectly_unary_invocable<projected<iterator_t<R>, Proj>> Fun>
//   constexpr ranges::for_each_result<borrowed_iterator_t<R>, Fun>
//     ranges::for_each(R&& r, Fun f, Proj proj = {});

#include <algorithm>
#include <array>
#include <cassert>
#include <deque>
#include <ranges>
#include <vector>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct Callable {
  void operator()(int);
};

template <class Iter, class Sent = Iter>
concept HasForEachIt = requires(Iter iter, Sent sent) { std::ranges::for_each(iter, sent, Callable{}); };

static_assert(HasForEachIt<int*>);
static_assert(!HasForEachIt<InputIteratorNotDerivedFrom>);
static_assert(!HasForEachIt<InputIteratorNotIndirectlyReadable>);
static_assert(!HasForEachIt<InputIteratorNotInputOrOutputIterator>);
static_assert(!HasForEachIt<int*, SentinelForNotSemiregular>);
static_assert(!HasForEachIt<int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Func>
concept HasForEachItFunc = requires(int* a, int* b, Func func) { std::ranges::for_each(a, b, func); };

static_assert(HasForEachItFunc<Callable>);
static_assert(!HasForEachItFunc<IndirectUnaryPredicateNotPredicate>);
static_assert(!HasForEachItFunc<IndirectUnaryPredicateNotCopyConstructible>);

template <class Range>
concept HasForEachR = requires(Range range) { std::ranges::for_each(range, Callable{}); };

static_assert(HasForEachR<UncheckedRange<int*>>);
static_assert(!HasForEachR<InputRangeNotDerivedFrom>);
static_assert(!HasForEachR<InputRangeNotIndirectlyReadable>);
static_assert(!HasForEachR<InputRangeNotInputOrOutputIterator>);
static_assert(!HasForEachR<InputRangeNotSentinelSemiregular>);
static_assert(!HasForEachR<InputRangeNotSentinelEqualityComparableWith>);

template <class Func>
concept HasForEachRFunc = requires(UncheckedRange<int*> a, Func func) { std::ranges::for_each(a, func); };

static_assert(HasForEachRFunc<Callable>);
static_assert(!HasForEachRFunc<IndirectUnaryPredicateNotPredicate>);
static_assert(!HasForEachRFunc<IndirectUnaryPredicateNotCopyConstructible>);

template <class Iter, class Sent = Iter>
constexpr void test_iterator() {
  { // simple test
    {
      auto func = [i = 0](int& a) mutable { a += i++; };
      int a[]   = {1, 6, 3, 4};
      std::same_as<std::ranges::for_each_result<Iter, decltype(func)>> decltype(auto) ret =
          std::ranges::for_each(Iter(a), Sent(Iter(a + 4)), func);
      assert(a[0] == 1);
      assert(a[1] == 7);
      assert(a[2] == 5);
      assert(a[3] == 7);
      assert(base(ret.in) == a + 4);
      int i = 0;
      ret.fun(i);
      assert(i == 4);
    }
    {
      auto func  = [i = 0](int& a) mutable { a += i++; };
      int a[]    = {1, 6, 3, 4};
      auto range = std::ranges::subrange(Iter(a), Sent(Iter(a + 4)));
      std::same_as<std::ranges::for_each_result<Iter, decltype(func)>> decltype(auto) ret =
          std::ranges::for_each(range, func);
      assert(a[0] == 1);
      assert(a[1] == 7);
      assert(a[2] == 5);
      assert(a[3] == 7);
      assert(base(ret.in) == a + 4);
      int i = 0;
      ret.fun(i);
      assert(i == 4);
    }
  }

  { // check that an empty range works
    {
      std::array<int, 0> a = {};
      std::ranges::for_each(Iter(a.data()), Sent(Iter(a.data())), [](auto&) { assert(false); });
    }
    {
      std::array<int, 0> a = {};
      auto range           = std::ranges::subrange(Iter(a.data()), Sent(Iter(a.data())));
      std::ranges::for_each(range, [](auto&) { assert(false); });
    }
  }
}

struct deque_test {
  std::deque<int>* d_;
  int* i_;

  deque_test(std::deque<int>& d, int& i) : d_(&d), i_(&i) {}

  void operator()(int& v) {
    assert(&(*d_)[*i_] == &v);
    ++*i_;
  }
};

/*TEST_CONSTEXPR_CXX26*/
void test_segmented_deque_iterator() { // TODO: Mark as TEST_CONSTEXPR_CXX26 once std::deque is constexpr
  // check that segmented deque iterators work properly
  int sizes[] = {0, 1, 2, 1023, 1024, 1025, 2047, 2048, 2049};
  for (const int size : sizes) {
    std::deque<int> d(size);
    int index = 0;

    std::ranges::for_each(d, deque_test(d, index));
  }
}

constexpr bool test() {
  test_iterator<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>();
  test_iterator<cpp20_input_iterator<int*>, sentinel_wrapper<cpp20_input_iterator<int*>>>();
  test_iterator<forward_iterator<int*>>();
  test_iterator<bidirectional_iterator<int*>>();
  test_iterator<random_access_iterator<int*>>();
  test_iterator<contiguous_iterator<int*>>();
  test_iterator<int*>();

  { // check that std::invoke is used
    struct S {
      int check;
      int other;
    };
    {
      S a[] = {{1, 2}, {3, 4}, {5, 6}};
      std::ranges::for_each(a, a + 3, [](int& i) { i = 0; }, &S::check);
      assert(a[0].check == 0);
      assert(a[0].other == 2);
      assert(a[1].check == 0);
      assert(a[1].other == 4);
      assert(a[2].check == 0);
      assert(a[2].other == 6);
    }
    {
      S a[] = {{1, 2}, {3, 4}, {5, 6}};
      std::ranges::for_each(a, [](int& i) { i = 0; }, &S::check);
      assert(a[0].check == 0);
      assert(a[0].other == 2);
      assert(a[1].check == 0);
      assert(a[1].other == 4);
      assert(a[2].check == 0);
      assert(a[2].other == 6);
    }
  }

  if (!TEST_IS_CONSTANT_EVALUATED) // TODO: Use TEST_STD_AT_LEAST_26_OR_RUNTIME_EVALUATED when std::deque is made constexpr
    test_segmented_deque_iterator();

  {
    std::vector<std::vector<int>> vec = {{0}, {1, 2}, {3, 4, 5}, {6, 7, 8, 9}, {10}, {11, 12, 13}};
    auto v                            = vec | std::views::join;
    std::ranges::for_each(v, [i = 0](int x) mutable { assert(x == 2 * i++); }, [](int x) { return 2 * x; });
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
