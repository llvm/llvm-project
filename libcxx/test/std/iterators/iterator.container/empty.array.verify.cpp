// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// <iterator>
// template <class T, size_t N> constexpr bool empty(const T (&array)[N]) noexcept;

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <vector>
#include <iterator>

#include "test_macros.h"

int main(int, char**)
{
    int c[5];
    std::empty(c); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    return 0;
}
