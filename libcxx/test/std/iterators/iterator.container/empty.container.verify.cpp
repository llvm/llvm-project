//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// <iterator>
// template <class C> constexpr auto empty(const C& c) -> decltype(c.empty());

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <vector>
#include <iterator>

void f() {
    std::vector<int> c;
    std::empty(c); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
