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

#include "min_allocator.h"

// Verify that std::indirect rejects array types.
void test() {
  // expected-error@*:* 2 {{static assertion failed}}
  std::indirect<int[]> i1;   // expected-note {{requested here}}
  std::indirect<int[10]> i2; // expected-note {{requested here}}
}
