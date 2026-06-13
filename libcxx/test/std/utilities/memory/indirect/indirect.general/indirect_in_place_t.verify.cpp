//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// template <class T, class Allocator = std::allocator<T>> class indirect;

#include <memory>

// Verify that std::indirect rejects in_place_t.
void test() {
  // expected-error@*:* 1 {{static assertion failed}}
  std::indirect<std::in_place_t> i1; // expected-note {{requested here}}
}
