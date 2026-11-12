//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// shared_ptr

// bool unique() const; // deprecated in C++17, removed in C++20

#include <memory>

void f() {
  const std::shared_ptr<int> p;
  p.unique(); // expected-error {{no member named 'unique' in 'std::shared_ptr<int>}}
}
