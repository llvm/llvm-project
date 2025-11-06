//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// Test the libc++ extension that std::indirect<T>::valueless_after_move is marked as [[nodiscard]].

#include <memory>

void test(std::indirect<int>& i) {
  i.valueless_after_move(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
