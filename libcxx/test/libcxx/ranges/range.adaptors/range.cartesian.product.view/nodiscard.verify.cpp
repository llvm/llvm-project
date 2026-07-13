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

// Non-simple: only the non-const begin()/end() overloads are viable.
struct NonSimpleView : std::ranges::view_base {
  int* begin();
  int* end();
};

// Simple: only the const begin()/end() overloads are viable.
struct ConstAccessibleView : std::ranges::view_base {
  int* begin() const;
  int* end() const;
};

// Distinct sentinel type: not a common range, so end() yields default_sentinel_t.
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

void test() {
  // [range.cartesian.view]

  std::ranges::cartesian_product_view<NonSimpleView> non_simple{NonSimpleView{}};
  non_simple.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  non_simple.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  const std::ranges::cartesian_product_view<ConstAccessibleView> const_view{ConstAccessibleView{}};
  const_view.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  const_view.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Not a common range: end() returns default_sentinel_t.
  const std::ranges::cartesian_product_view<NonCommonView> non_common{NonCommonView{}};
  non_common.end(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::ranges::cartesian_product_view<SizedView> sized{SizedView{}};
  sized.size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(sized).size();

  // [range.cartesian.iterator]

  const std::ranges::cartesian_product_view<ConstAccessibleView> view{ConstAccessibleView{}};
  auto iter = view.begin();

  *iter;    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter++;   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter--;   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter[0];  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter + 0; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  0 + iter; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter - 0; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // operator== and operator<=> are omitted: -Wunused-value already warns on discarded comparisons.

  iter - iter; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  iter - std::default_sentinel;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::default_sentinel - iter;

  iter_move(iter); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // [range.cartesian.overview]
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::cartesian_product();
}
