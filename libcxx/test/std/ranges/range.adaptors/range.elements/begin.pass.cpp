//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

//  constexpr auto begin() requires (!simple-view<V>)
//  constexpr auto begin() const requires range<const V>

#include <cassert>
#include <concepts>
#include <ranges>
#include <tuple>
#include <utility>

#include "types.h"

template <class T>
concept HasConstBegin = requires(const T ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T t) { t.begin(); };

template <class T>
concept HasConstAndNonConstBegin =
    HasConstBegin<T> &&
    // because const begin and non-const begin returns different types (iterator<true>, iterator<false>)
    requires(T t, const T ct) { requires !std::same_as<decltype(t.begin()), decltype(ct.begin())>; };

template <class T>
concept HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
concept HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

struct NoConstBeginView : TupleBufferView {
  using TupleBufferView::TupleBufferView;
  constexpr std::tuple<int>* begin() { return buffer_; }
  constexpr std::tuple<int>* end() { return buffer_ + size_; }
};

// simple-view<V>
static_assert(HasOnlyConstBegin<std::ranges::elements_view<SimpleCommon, 0>>);

// !simple-view<V> && range<const V>
static_assert(HasConstAndNonConstBegin<std::ranges::elements_view<NonSimpleCommon, 0>>);

// !range<const V>
static_assert(HasOnlyNonConstBegin<std::ranges::elements_view<NoConstBeginView, 0>>);

constexpr bool test() {
  std::tuple<int> buffer[] = {{1}, {2}};
  {
    // underlying iterator should be pointing to the first element
    auto ev   = std::views::elements<0>(buffer);
    auto iter = ev.begin();
    assert(&(*iter) == &std::get<0>(buffer[0]));
  }

  {
    // underlying range models simple-view
    auto v = std::views::elements<0>(SimpleCommon{buffer});
    static_assert(std::is_same_v<decltype(v.begin()), decltype(std::as_const(v).begin())>);
    assert(v.begin() == std::as_const(v).begin());
    auto&& r = *std::as_const(v).begin();
    assert(&r == &std::get<0>(buffer[0]));
  }

  {
    // underlying const R is not a range
    auto v   = std::views::elements<0>(NoConstBeginView{buffer});
    auto&& r = *v.begin();
    assert(&r == &std::get<0>(buffer[0]));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
