//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto end() requires (!simple-view<V>)
// { return sentinel<false>(ranges::end(base_), addressof(*pred_)); }
// constexpr auto end() const
//   requires range<const V> &&
//            indirect_unary_predicate<const Pred, iterator_t<const V>>
// { return sentinel<true>(ranges::end(base_), addressof(*pred_)); }

#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "types.h"

// Test Constraints
template <class T>
concept HasConstEnd = requires(const T& ct) { ct.end(); };

template <class T>
concept HasEnd = requires(T& t) { t.end(); };

template <class T>
concept HasConstAndNonConstEnd =
    HasConstEnd<T> && requires(T& t, const T& ct) { requires !std::same_as<decltype(t.end()), decltype(ct.end())>; };

template <class T>
concept HasOnlyNonConstEnd = HasEnd<T> && !HasConstEnd<T>;

template <class T>
concept HasOnlyConstEnd = HasConstEnd<T> && !HasConstAndNonConstEnd<T>;

struct Pred {
  constexpr bool operator()(int i) const { return i < 5; }
};

static_assert(HasOnlyConstEnd<std::ranges::take_while_view<SimpleView, Pred>>);

static_assert(HasOnlyNonConstEnd<std::ranges::take_while_view<ConstNotRange, Pred>>);

static_assert(HasConstAndNonConstEnd<std::ranges::take_while_view<NonSimple, Pred>>);

struct NotPredForConst {
  constexpr bool operator()(int& i) const { return i > 5; }
};
static_assert(HasOnlyNonConstEnd<std::ranges::take_while_view<NonSimple, NotPredForConst>>);

constexpr bool test() {
  // simple-view
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    SimpleView v{buffer};
    std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer + 4);
    decltype(auto) it2 = std::as_const(twv).end();
    assert(it2 == buffer + 4);

    static_assert(std::same_as<decltype(it1), decltype(it2)>);
  }

  // const not range
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    ConstNotRange v{buffer};
    std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer + 4);
  }

  // NonSimple
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    std::ranges::take_while_view twv(v, Pred{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer + 4);
    decltype(auto) it2 = std::as_const(twv).end();
    assert(it2 == buffer + 4);

    static_assert(!std::same_as<decltype(it1), decltype(it2)>);
  }

  // NotPredForConst
  // LWG 3450: The const overloads of `take_while_view::begin/end` are underconstrained
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    std::ranges::take_while_view twv(v, NotPredForConst{});
    decltype(auto) it1 = twv.end();
    assert(it1 == buffer);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
