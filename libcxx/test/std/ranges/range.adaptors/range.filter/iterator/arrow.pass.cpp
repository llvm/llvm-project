//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test is hitting Clang bugs with LSV in older versions of Clang.
// UNSUPPORTED: clang-modules-build && (clang-15 || apple-clang-14)

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator_t<V> operator->() const
//    requires has-arrow<iterator_t<V>> && copyable<iterator_t<V>>

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

struct XYPoint {
  int x;
  int y;
};

template <class T>
concept has_arrow = requires (T t) {
  { t->x };
};
static_assert(has_arrow<XYPoint*>); // test the test

struct WithArrowOperator {
  using iterator_category = std::input_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = XYPoint;

  constexpr explicit WithArrowOperator(XYPoint* p) : p_(p) { }
  constexpr XYPoint& operator*() const { return *p_; }
  constexpr XYPoint* operator->() const { return p_; } // has arrow
  constexpr WithArrowOperator& operator++() { ++p_; return *this; }
  constexpr WithArrowOperator operator++(int) { return WithArrowOperator(p_++); }

  friend constexpr XYPoint* base(WithArrowOperator const& i) { return i.p_; }
  XYPoint* p_;
};
static_assert(std::input_iterator<WithArrowOperator>);

struct WithNonCopyableIterator : std::ranges::view_base {
  struct iterator {
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = XYPoint;

    iterator(iterator const&) = delete; // not copyable
    iterator(iterator&&);
    iterator& operator=(iterator&&);
    XYPoint& operator*() const;
    iterator operator->() const;
    iterator& operator++();
    iterator operator++(int);

    // We need this to use XYPoint* as a sentinel type below. sentinel_wrapper
    // can't be used because this iterator is not copyable.
    friend bool operator==(iterator const&, XYPoint*);
  };

  iterator begin() const;
  XYPoint* end() const;
};
static_assert(std::ranges::input_range<WithNonCopyableIterator>);

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  std::array<XYPoint, 5> array{{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}}};
  using View = minimal_view<Iterator, Sentinel>;
  using FilterView = std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = std::ranges::iterator_t<FilterView>;

  auto make_filter_view = [](auto begin, auto end, auto pred) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return FilterView(std::move(view), pred);
  };

  for (std::ptrdiff_t n = 0; n != 5; ++n) {
    FilterView view = make_filter_view(array.begin(), array.end(), AlwaysTrue{});
    FilterIterator const iter(view, Iterator(array.begin() + n));
    std::same_as<Iterator> decltype(auto) result = iter.operator->();
    assert(base(result) == array.begin() + n);
    assert(iter->x == n);
    assert(iter->y == n);
  }
}

constexpr bool tests() {
  test<WithArrowOperator>();
  test<XYPoint*>();
  test<XYPoint const*>();
  test<contiguous_iterator<XYPoint*>>();
  test<contiguous_iterator<XYPoint const*>>();

  // Make sure filter_view::iterator doesn't have operator-> if the
  // underlying iterator doesn't have one.
  {
    auto check_no_arrow = []<class It> {
      using View = minimal_view<It, sentinel_wrapper<It>>;
      using FilterView = std::ranges::filter_view<View, AlwaysTrue>;
      using FilterIterator = std::ranges::iterator_t<FilterView>;
      static_assert(!has_arrow<FilterIterator>);
    };
    check_no_arrow.operator()<cpp17_input_iterator<XYPoint*>>();
    check_no_arrow.operator()<cpp20_input_iterator<XYPoint*>>();
    check_no_arrow.operator()<forward_iterator<XYPoint*>>();
    check_no_arrow.operator()<bidirectional_iterator<XYPoint*>>();
    check_no_arrow.operator()<random_access_iterator<XYPoint*>>();
    check_no_arrow.operator()<int*>();
  }

  // Make sure filter_view::iterator doesn't have operator-> if the
  // underlying iterator is not copyable.
  {
    using FilterView = std::ranges::filter_view<WithNonCopyableIterator, AlwaysTrue>;
    using FilterIterator = std::ranges::iterator_t<FilterView>;
    static_assert(!has_arrow<FilterIterator>);
  }
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
