//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Test the libc++ extension that std::views::transform is marked as [[nodiscard]].

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

  auto identity_view = TestView{} | std::views::transform(std::identity{});
    
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  identity_view.begin();

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::transform(f);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::transform(range, f);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::views::transform(f);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::all | std::views::transform(f);
}
