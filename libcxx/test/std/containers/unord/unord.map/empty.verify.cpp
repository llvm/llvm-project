//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// class unordered_map

// bool empty() const noexcept;

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <unordered_map>

void f() {
    std::unordered_map<int, int> c;
    c.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
