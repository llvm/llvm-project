//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// allocator:
// T* allocate(size_t n, const void* hint);

// Removed in C++20.

#include <memory>

void f() {
  std::allocator<int> a;
  a.allocate(3, nullptr); // expected-error {{too many arguments to function call}}
}
