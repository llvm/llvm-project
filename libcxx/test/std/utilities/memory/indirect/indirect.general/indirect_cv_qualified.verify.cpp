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

// Verify that std::indirect rejects cv-qualified types.
void test() {
  // Use bare_allocator, because std::allocator doesn't support cv-qualified types.
  // expected-error@*:* 3 {{static assertion failed}}
  std::indirect<const int, bare_allocator<const int>> i1;                   // expected-note {{requested here}}
  std::indirect<volatile int, bare_allocator<volatile int>> i2;             // expected-note {{requested here}}
  std::indirect<const volatile int, bare_allocator<const volatile int>> i3; // expected-note {{requested here}}
}
