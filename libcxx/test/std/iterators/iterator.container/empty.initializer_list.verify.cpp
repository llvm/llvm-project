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
// template <class E> constexpr bool empty(initializer_list<E> il) noexcept;

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// REQUIRES: verify-support

#include <initializer_list>
#include <iterator>

#include "test_macros.h"

int main(int, char**)
{
    std::initializer_list<int> c = { 4 };
    std::empty(c); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    return 0;
}
