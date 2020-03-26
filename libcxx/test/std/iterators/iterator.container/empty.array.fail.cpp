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

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-3.3, clang-3.4, clang-3.5, clang-3.6, clang-3.7, clang-3.8
// REQUIRES: verify-support

#include <vector>
#include <iterator>

#include "test_macros.h"

int main(int, char**)
{
    int c[5];
    std::empty(c); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    return 0;
}
