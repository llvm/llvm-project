//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// constexpr auto begin() requires (!simple-view<V>);
// constexpr auto begin() const requires range-with-movable-references<const V>;

#include <cassert>
#include <concepts>
#include <ranges>

#include "test_iterators.h"

#include "types.h"

// Types

template <bool Simple>
struct CommonView : std::ranges::view_base {
  constexpr std::tuple<std::ptrdiff_t, int>* begin()
    requires(!Simple)
  {
    return nullptr;
  }
  constexpr const std::tuple<std::ptrdiff_t, int>* begin() const { return nullptr; }
  constexpr std::tuple<std::ptrdiff_t, int>* end()
    requires(!Simple)
  {
    return nullptr;
  }
  constexpr const std::tuple<std::ptrdiff_t, int>* end() const { return nullptr; }
};
using SimpleCommonView    = CommonView<true>;
using NonSimpleCommonView = CommonView<false>;

struct NoConstBeginView : std::ranges::view_base {
  constexpr std::tuple<std::ptrdiff_t, int>* begin() { return nullptr; }
  constexpr std::tuple<std::ptrdiff_t, int>* end() { return nullptr; }
};

// SFINAE

template <class T>
concept HasConstBegin = requires(const T ct) { ct.begin(); };

template <class T>
concept HasBegin = requires(T t) { t.begin(); };

template <class T>
concept HasConstAndNonConstBegin =
    HasConstBegin<T> &&
    // Because const begin() and non-const begin() returns different types: iterator<true> vs. iterator<false>
    requires(T t, const T ct) { requires !std::same_as<decltype(t.begin()), decltype(ct.begin())>; };

template <class T>
concept HasOnlyNonConstBegin = HasBegin<T> && !HasConstBegin<T>;

template <class T>
concept HasOnlyConstBegin = HasConstBegin<T> && !HasConstAndNonConstBegin<T>;

// simple-view<V>
static_assert(HasOnlyConstBegin<std::ranges::enumerate_view<SimpleCommonView>>);

// !simple-view<V> && range<const V>
static_assert(HasConstAndNonConstBegin<std::ranges::enumerate_view<NonSimpleCommonView>>);

// !range<const V>
static_assert(HasOnlyNonConstBegin<std::ranges::enumerate_view<NoConstBeginView>>);

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Check the return type of begin()
  {
    RangeView range(buff, buff + 1);

    std::ranges::enumerate_view view(range);
    using Iterator = std::ranges::iterator_t<decltype(view)>;
    static_assert(std::same_as<Iterator, decltype(view.begin())>);
    // static_assert(std::same_as<ValueType<int>, decltype(*view.begin())>);
  }

  // begin() over an empty range
  {
    RangeView range(buff, buff);

    std::ranges::enumerate_view view(range);
    auto it = view.begin();
    assert(base(it.base()) == buff);
    assert(it == view.end());
  }

  // begin() over a 1-element range
  {
    RangeView range(buff, buff + 1);

    std::ranges::enumerate_view view(range);
    auto it = view.begin();
    assert(base(it.base()) == buff);
  }

  // begin() over an N-element range
  {
    RangeView range(buff, buff + 8);

    std::ranges::enumerate_view view(range);
    auto it = view.begin();
    assert(base(it.base()) == buff);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
