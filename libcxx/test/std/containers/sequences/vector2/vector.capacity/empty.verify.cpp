//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <vector>

// class vector2

// bool empty() const noexcept;


#include <vector>

void f() {
  std::vector2<int> c;
  c.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
