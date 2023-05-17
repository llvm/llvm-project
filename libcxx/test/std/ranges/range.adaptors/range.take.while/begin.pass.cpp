//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr auto begin() requires (!simple-view<V>)
// { return ranges::begin(base_); }
//
// constexpr auto begin() const
//   requires range<const V> &&
//            indirect_unary_predicate<const Pred, iterator_t<const V>>
// { return ranges::begin(base_); }

#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "types.h"

// Test Constraints
template <class T>
concept HasConstBegin = requires(const T& ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T& t) { t.begin(); };

template <class T>
concept HasConstAndNonConstBegin =
    HasConstBegin<T> &&
    requires(T& t, const T& ct) { requires !std::same_as<decltype(t.begin()), decltype(ct.begin())>; };

template <class T>
concept HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
concept HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

struct Pred {
  constexpr bool operator()(int i) const { return i > 5; }
};

static_assert(HasOnlyConstBegin<std::ranges::take_while_view<SimpleView, Pred>>);

static_assert(HasOnlyNonConstBegin<std::ranges::take_while_view<ConstNotRange, Pred>>);

static_assert(HasConstAndNonConstBegin<std::ranges::take_while_view<NonSimple, Pred>>);

struct NotPredForConst {
  constexpr bool operator()(int& i) const { return i > 5; }
};
static_assert(HasOnlyNonConstBegin<std::ranges::take_while_view<NonSimple, NotPredForConst>>);

constexpr bool test() {
  // simple-view
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    SimpleView v{buffer};
    std::ranges::take_while_view twv(v, Pred{});
    std::same_as<int*> decltype(auto) it1 = twv.begin();
    assert(it1 == buffer);
    std::same_as<int*> decltype(auto) it2 = std::as_const(twv).begin();
    assert(it2 == buffer);
  }

  // const not range
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    ConstNotRange v{buffer};
    std::ranges::take_while_view twv(v, Pred{});
    std::same_as<int*> decltype(auto) it1 = twv.begin();
    assert(it1 == buffer);
  }

  // NonSimple
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    std::ranges::take_while_view twv(v, Pred{});
    std::same_as<int*> decltype(auto) it1 = twv.begin();
    assert(it1 == buffer);
    std::same_as<const int*> decltype(auto) it2 = std::as_const(twv).begin();
    assert(it2 == buffer);
  }

  // NotPredForConst
  // LWG 3450: The const overloads of `take_while_view::begin/end` are underconstrained
  {
    int buffer[] = {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    NonSimple v{buffer};
    std::ranges::take_while_view twv(v, NotPredForConst{});
    std::same_as<int*> decltype(auto) it1 = twv.begin();
    assert(it1 == buffer);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
