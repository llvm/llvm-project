//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto begin()
// constexpr auto begin() const

#include <array>
#include <cassert>
#include <concepts>
#include <ranges>

#include "test_iterators.h"

struct SimpleView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct NonSimpleView : std::ranges::view_base {
  char* begin();
  char* end();
  int* begin() const;
  int* end() const;
};

struct NonConstView : std::ranges::view_base {
  char* begin();
  char* end();
};

template <class T>
concept HasBegin = requires(T t) { t.begin(); };

static_assert(HasBegin<std::ranges::as_rvalue_view<SimpleView>>);
static_assert(HasBegin<const std::ranges::as_rvalue_view<SimpleView>>);
static_assert(HasBegin<std::ranges::as_rvalue_view<NonSimpleView>>);
static_assert(HasBegin<const std::ranges::as_rvalue_view<NonSimpleView>>);
static_assert(HasBegin<std::ranges::as_rvalue_view<NonConstView>>);
static_assert(!HasBegin<const std::ranges::as_rvalue_view<NonConstView>>);

template <class Iter, class Sent>
constexpr void test_range() {
  int a[] = {1, 2};
  std::ranges::subrange range(Iter(std::begin(a)), Sent(Iter(std::end(a))));
  std::ranges::as_rvalue_view view(std::move(range));
  std::same_as<std::move_iterator<Iter>> decltype(auto) iter = view.begin();
  assert(base(iter.base()) == std::begin(a));
}

template <class Iter, class Sent>
class WrapRange {
  Iter iter_;
  Sent sent_;

public:
  constexpr WrapRange(Iter iter, Sent sent) : iter_(std::move(iter)), sent_(std::move(sent)) {}

  constexpr Iter begin() const { return iter_; }
  constexpr Sent end() const { return sent_; }
};

template <class Iter, class Sent>
WrapRange(Iter, Sent) -> WrapRange<Iter, Sent>;

template <class Iter, class Sent>
constexpr void test_const_range() {
  int a[]    = {1, 2};
  auto range = WrapRange{Iter(a), Sent(Iter(a + 2))};
  const std::ranges::as_rvalue_view view(std::views::all(range));
  std::same_as<std::move_iterator<Iter>> decltype(auto) iter = view.begin();
  assert(base(iter.base()) == std::begin(a));
}

struct move_iterator_view : std::ranges::view_base {
  constexpr std::move_iterator<int*> begin() const { return {}; }
  constexpr std::move_iterator<int*> end() const { return {}; }
};

constexpr bool test() {
  meta::for_each(meta::cpp20_input_iterator_list<int*>{}, []<class Iter> {
    if constexpr (std::sentinel_for<Iter, Iter>)
      test_range<Iter, Iter>();
    test_range<Iter, sentinel_wrapper<Iter>>();
    test_range<Iter, sized_sentinel<Iter>>();
  });

  meta::for_each(meta::forward_iterator_list<const int*>{}, []<class Iter> {
    test_const_range<Iter, Iter>();
    test_const_range<Iter, sentinel_wrapper<Iter>>();
    test_const_range<Iter, sized_sentinel<Iter>>();
  });

  { // check that with a std::move_iterator begin() doesn't return move_iterator<move_iterator<T>>
    std::ranges::as_rvalue_view view{move_iterator_view{}};
    std::same_as<std::move_iterator<int*>> decltype(auto) it = view.begin();
    assert(it == std::move_iterator<int*>{});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
