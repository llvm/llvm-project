//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <ranges>

// class cache_latest_view

//    constexpr auto end();

#include <cassert>
#include <concepts>
#include <ranges>

#include "test_iterators.h"
#include <print>

struct NonConstView : std::ranges::view_base {
  int* begin();
  int* end();
};

struct ConstView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

struct ConstNonConstView : std::ranges::view_base {
  int* begin();
  int* end();
  int* begin() const;
  int* end() const;
};

template <class T>
concept HasEnd = requires(T t) { t.end(); };

static_assert(HasEnd<std::ranges::cache_latest_view<NonConstView>>);
static_assert(!HasEnd<const std::ranges::cache_latest_view<NonConstView>>);
static_assert(HasEnd<std::ranges::cache_latest_view<ConstView>>);
static_assert(!HasEnd<const std::ranges::cache_latest_view<ConstView>>);
static_assert(HasEnd<std::ranges::cache_latest_view<ConstNonConstView>>);
static_assert(!HasEnd<const std::ranges::cache_latest_view<ConstNonConstView>>);

struct CommonView : std::ranges::view_base {
  int i_;

  constexpr forward_iterator<int*> begin() { return forward_iterator<int*>(&i_); }
  constexpr forward_iterator<int*> end() { return begin(); }
};

static_assert(std::ranges::common_range<CommonView>);
static_assert(std::ranges::forward_range<CommonView>);
static_assert(std::ranges::input_range<CommonView>);

struct NonCommonView : std::ranges::view_base {
  int i_;

  constexpr forward_iterator<int*> begin() { return forward_iterator<int*>(&i_); }
  constexpr sentinel_wrapper<forward_iterator<int*>> end() { return sentinel_wrapper<forward_iterator<int*>>(begin()); }
};

static_assert(!std::ranges::common_range<NonCommonView>);
static_assert(std::ranges::forward_range<NonCommonView>);
static_assert(std::ranges::input_range<NonCommonView>);
static_assert(
    std::derived_from<typename std::iterator_traits<std::ranges::iterator_t<NonCommonView>>::iterator_category,
                      std::input_iterator_tag>);

constexpr bool test() {
  {
    CommonView view{{}, 94};

    std::same_as<std::ranges::cache_latest_view<CommonView>> decltype(auto) v = view | std::views::cache_latest;

    assert(*base(v.end().base()) == view.i_);
  }
  {
    NonCommonView view{{}, 94};

    std::same_as<std::ranges::cache_latest_view<NonCommonView>> decltype(auto) v = view | std::views::cache_latest;

    assert(*base(v.end().base()) == view.i_);
  }

  {
    int arr[] = {82, 94, 76};

    std::same_as<std::ranges::cache_latest_view<std::ranges::ref_view<int[3]>>> decltype(auto) v =
        arr | std::views::cache_latest;

    assert(base(v.end().base()) == arr + 3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
