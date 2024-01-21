//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<input_iterator I, class Proj = identity,
//          indirectly_unary_invocable<projected<I, Proj>> Fun>
//   constexpr ranges::for_each_n_result<I, Fun>
//     ranges::for_each_n(I first, iter_difference_t<I> n, Fun f, Proj proj = {});

#include <algorithm>
#include <array>
#include <ranges>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

struct Callable {
  void operator()(int);
};

template <class Iter>
concept HasForEachN = requires (Iter iter) { std::ranges::for_each_n(iter, 0, Callable{}); };

static_assert(HasForEachN<int*>);
static_assert(!HasForEachN<InputIteratorNotDerivedFrom>);
static_assert(!HasForEachN<InputIteratorNotIndirectlyReadable>);
static_assert(!HasForEachN<InputIteratorNotInputOrOutputIterator>);

template <class Func>
concept HasForEachItFunc = requires(int* a, int b, Func func) { std::ranges::for_each_n(a, b, func); };

static_assert(HasForEachItFunc<Callable>);
static_assert(!HasForEachItFunc<IndirectUnaryPredicateNotPredicate>);
static_assert(!HasForEachItFunc<IndirectUnaryPredicateNotCopyConstructible>);

template <class Iter>
constexpr void test_iterator() {
  { // simple test
    auto func = [i = 0](int& a) mutable { a += i++; };
    int a[] = {1, 6, 3, 4};
    std::same_as<std::ranges::for_each_result<Iter, decltype(func)>> auto ret =
        std::ranges::for_each_n(Iter(a), 4, func);
    assert(a[0] == 1);
    assert(a[1] == 7);
    assert(a[2] == 5);
    assert(a[3] == 7);
    assert(base(ret.in) == a + 4);
    int i = 0;
    ret.fun(i);
    assert(i == 4);
  }

  { // check that an empty range works
    std::array<int, 0> a = {};
    std::ranges::for_each_n(Iter(a.data()), 0, [](auto&) { assert(false); });
  }
}

constexpr bool test() {
  test_iterator<cpp17_input_iterator<int*>>();
  test_iterator<cpp20_input_iterator<int*>>();
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

    S a[] = {{1, 2}, {3, 4}, {5, 6}};
    std::ranges::for_each_n(a, 3, [](int& i) { i = 0; }, &S::check);
    assert(a[0].check == 0);
    assert(a[0].other == 2);
    assert(a[1].check == 0);
    assert(a[1].other == 4);
    assert(a[2].check == 0);
    assert(a[2].other == 6);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
