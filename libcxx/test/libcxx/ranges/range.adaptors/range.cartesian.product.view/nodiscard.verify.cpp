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
#include <utility>

// Tests non-const begin()/end() overloads. Returns iterator from end().
struct NonSimpleNonCommonView : std::ranges::view_interface<NonSimpleNonCommonView> {
  int* begin();
  const int* begin() const;
  volatile int* end();
  const volatile int* end() const;
  unsigned size() const;
};
static_assert(!std::same_as<std::ranges::iterator_t<NonSimpleNonCommonView>,
                            std::ranges::iterator_t<const NonSimpleNonCommonView>>);
static_assert(!std::same_as<std::ranges::sentinel_t<NonSimpleNonCommonView>,
                            std::ranges::sentinel_t<const NonSimpleNonCommonView>>);
static_assert(!std::ranges::common_range<NonSimpleNonCommonView> &&
              !std::ranges::common_range<const NonSimpleNonCommonView>);
static_assert(std::ranges::sized_range<NonSimpleNonCommonView> &&
              std::ranges::random_access_range<NonSimpleNonCommonView>);
static_assert(std::ranges::sized_range<const NonSimpleNonCommonView> &&
              std::ranges::random_access_range<const NonSimpleNonCommonView>);

// Non-sized and random access range so end() yields default_sentinel_t.
struct NonCommonView : std::ranges::view_base {
  struct Sentinel {
    friend bool operator==(int*, Sentinel) noexcept;
  };
  int* begin() const;
  Sentinel end() const;
};
static_assert(!std::ranges::common_range<const NonCommonView>);
static_assert(!std::ranges::sized_range<const NonCommonView>);

void test() {
  // [range.cartesian.view]

  std::ranges::cartesian_product_view<NonSimpleNonCommonView> v{NonSimpleNonCommonView{}};
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  v.size();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).begin();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).end();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(v).size();

  const std::ranges::cartesian_product_view<NonCommonView> non_common{NonCommonView{}};
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  non_common.end();

  // [range.cartesian.iterator]

  auto it   = v.begin();
  auto c_it = std::as_const(v).begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  *c_it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  c_it[0];
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it + 0;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  0 + it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - 0;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - it;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it - std::default_sentinel;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::default_sentinel - it;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter_move(c_it);

  // [range.cartesian.overview]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::cartesian_product();

  int range[] = {1, 2, 3};
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::cartesian_product(range);
}
