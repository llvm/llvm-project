//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// Test the libc++ extension that std::indirect's observers are marked as [[nodiscard]].

#include <memory>
#include <utility>

void test(std::indirect<int>& i) {
  // clang-format off
  *i; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::move(i); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::as_const(i); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::move(std::as_const(i)); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  i.valueless_after_move(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  i.get_allocator(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  i.operator->(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::as_const(i).operator->(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on
}
