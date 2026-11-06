//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:
// size_type max_size() const throw();

// In C++20, parts of std::allocator<T> have been removed.
// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <memory>

void test(const std::allocator<int>& a) {
  (void)a.max_size(); // expected-error {{no member}}
}
