//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// Test the libc++ extension that std::ranges::transform_view and std::views::transform are marked as [[nodiscard]].

#include <ranges>
#include <functional>

struct TestView : std::ranges::view_interface<TestView> {
  int* begin();
  char* begin() const;
  const int* end();
  const char* end() const;
};

void test() {
  int range[] = {1, 2, 3};
  auto f      = [](int i) { return i; };

  auto identity_view     = TestView{} | std::views::transform(std::identity{});
  auto transformed_range = range | std::views::transform(f);

  const auto const_identity_view     = TestView{} | std::views::transform(std::identity{});
  const auto const_transformed_range = range | std::views::transform(f);

  auto it             = identity_view.begin();
  const auto const_it = identity_view.begin();

  auto transformed_range_sent             = transformed_range.end();
  auto identity_view_sent                 = identity_view.end();
  const auto const_transformed_range_sent = transformed_range.end();
  const auto const_identity_view_sent     = identity_view.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  const_it.base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  it.base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::transform(identity_view, fn).base();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  transformed_range.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  const_transformed_range.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  transformed_range.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  identity_view.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  const_transformed_range.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  const_identity_view.end();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::transform(f);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::transform(range, f);

// expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  const_it[0];

  {
    // ===== Non-const views =====

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it + 1;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    1 + it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - 1;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it - it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    it[0];
  }

  {

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    const_it + 1;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    1 + const_it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    const_it - 1;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    const_it - const_it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    *const_it;

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    const_it[0];
  }

  {
    // ===== Non-const sentinels =====
    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    transformed_range_sent.base();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    identity_view_sent.base();
  }

  {
    // ===== Const sentinels =====

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    const_transformed_range_sent.base();

    // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
    const_identity_view_sent.base();
  }
}
