//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <functional>

// Test the libc++ extension that std::bind_front<NTTP> is marked as [[nodiscard]].

#include <functional>

void test() {
  std::bind_front<test>(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
