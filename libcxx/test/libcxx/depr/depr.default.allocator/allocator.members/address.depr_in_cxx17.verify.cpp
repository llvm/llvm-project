//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:
// pointer address(reference x) const;
// const_pointer address(const_reference x) const;

// Deprecated in C++17

// REQUIRES: c++17

#include <memory>

void f() {
  int x = 0;
  std::allocator<int> a;

  (void)a.address(x); // expected-warning {{'address' is deprecated}}
}
