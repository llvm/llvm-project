//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// #include <memory>

// template<size_t N, class T>
// [[nodiscard]] constexpr T* assume_aligned(T* ptr);

#include <memory>

void f() {
  int *p = nullptr;
  std::assume_aligned<4>(p); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
