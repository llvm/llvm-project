//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// Check that functions are marked [[nodiscard]]

#include <ranges>

struct NonSimpleView : std::ranges::view_base {
  int* begin();
  int* end();
};

struct ConstAccessibleView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

// A view with a distinct sentinel type so __cartesian_product_is_common is false,
// causing end() to return default_sentinel_t.
struct NonCommonView : std::ranges::view_base {
  struct Sentinel {
    friend bool operator==(int*, Sentinel) noexcept;
  };
  int* begin() const;
  Sentinel end() const;
};

struct SizedView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
  unsigned size() const;
};

// testing cartesian_product_view
void test_view() {
  { // begin() requires (!__simple_view<First> || ... || !__simple_view<Vs>)
    static_assert(!std::ranges::__simple_view<NonSimpleView>);
    std::ranges::cartesian_product_view<NonSimpleView> view{NonSimpleView{}};
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    view.begin();
  }

  { // begin() const requires (range<const First> && ... && range<const Vs>)
    static_assert(std::ranges::range<const ConstAccessibleView>);
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    view.begin();
  }

  { // end() requires (!__simple_view<First> || ... || !__simple_view<Vs>) && __cartesian_product_is_common<First, Vs...>
    static_assert(!std::ranges::__simple_view<NonSimpleView>);
    static_assert(std::ranges::__cartesian_product_common_arg<NonSimpleView>);
    std::ranges::cartesian_product_view<NonSimpleView> view{NonSimpleView{}};
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    view.end();
  }

  { // end() const requires __cartesian_product_is_common<const First, const Vs...>
    static_assert(std::ranges::__cartesian_product_common_arg<const ConstAccessibleView>);
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    view.end();
  }

  { // end() const noexcept -> default_sentinel_t
    // Selected when neither end() overload applies, i.e. __cartesian_product_is_common is false.
    static_assert(!std::ranges::__cartesian_product_common_arg<const NonCommonView>);
    const std::ranges::cartesian_product_view<NonCommonView> view{NonCommonView{}};
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    view.end();
  }

  { // size() requires __cartesian_product_is_sized<First, Vs...>
    static_assert(std::ranges::__cartesian_product_is_sized<SizedView>);
    std::ranges::cartesian_product_view<SizedView> view{SizedView{}};
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    view.size();
  }

  { // size() const requires __cartesian_product_is_sized<const First, const Vs...>
    static_assert(std::ranges::__cartesian_product_is_sized<const SizedView>);
    const std::ranges::cartesian_product_view<SizedView> view{SizedView{}};
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    view.size();
  }
}

// testing cartesian_product_view::iterator
void test_iterator() {
  { // operator*() const
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    const auto iter = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *iter;
  }

  { // operator++(int) requires forward_range<const First>
    static_assert(std::ranges::forward_range<const ConstAccessibleView>);
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    auto iter = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter++;
  }

  { // operator--(int) requires __cartesian_product_is_bidirectional<Const, First, Vs...>
    static_assert(std::ranges::__cartesian_product_is_bidirectional<true, ConstAccessibleView>);
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    auto iter = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter--;
  }

  { // operator[](difference_type) const requires __cartesian_product_is_random_access<Const, First, Vs...>
    static_assert(std::ranges::__cartesian_product_is_random_access<true, ConstAccessibleView>);
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    const auto iter = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter[0];
  }

  // Note: comparison operators (operator==, operator<=>) are not tested here -- the compiler already
  // warns on discarded comparisons via -Wunused-value, independently of [[nodiscard]].

  { // friend constexpr iterator operator+(const iterator&, difference_type)
    // requires __cartesian_product_is_random_access<Const, First, Vs...>
    static_assert(std::ranges::__cartesian_product_is_random_access<true, ConstAccessibleView>);
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    const auto iter = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter + 0;
  }

  { // friend constexpr iterator operator+(difference_type, const iterator&)
    // requires __cartesian_product_is_random_access<Const, First, Vs...>
    static_assert(std::ranges::__cartesian_product_is_random_access<true, ConstAccessibleView>);
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    const auto iter = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    0 + iter;
  }

  { // friend constexpr iterator operator-(const iterator&, difference_type)
    // requires __cartesian_product_is_random_access<Const, First, Vs...>
    static_assert(std::ranges::__cartesian_product_is_random_access<true, ConstAccessibleView>);
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    const auto iter = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter - 0;
  }

  { // friend constexpr difference_type operator-(const iterator&, const iterator&)
    // requires __cartesian_is_sized_sentinel<Const, iterator_t, First, Vs...>
    static_assert(std::ranges::__cartesian_is_sized_sentinel<true, std::ranges::iterator_t, ConstAccessibleView>);
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    const auto iter = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter - iter;
  }

  { // friend constexpr difference_type operator-(const iterator&, default_sentinel_t)
    // requires __cartesian_is_sized_sentinel<Const, sentinel_t, First, Vs...>
    static_assert(std::ranges::__cartesian_is_sized_sentinel<true, std::ranges::sentinel_t, ConstAccessibleView>);
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    const auto iter = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter - std::default_sentinel;
  }

  { // friend constexpr difference_type operator-(default_sentinel_t, const iterator&)
    // requires __cartesian_is_sized_sentinel<Const, sentinel_t, First, Vs...>
    static_assert(std::ranges::__cartesian_is_sized_sentinel<true, std::ranges::sentinel_t, ConstAccessibleView>);
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    const auto iter = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::default_sentinel - iter;
  }

  { // iter_move(const iterator&)
    const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
    const auto iter = view.begin();
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    iter_move(iter);
  }
}

void test() {
  test_view();
  test_iterator();
}
