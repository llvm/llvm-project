// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// class array

// bool empty() const noexcept;

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-3.3, clang-3.4, clang-3.5, clang-3.6, clang-3.7, clang-3.8

#include <array>

#include "test_macros.h"

int main(int, char**)
{

    std::array<int, 1> c;
    c.empty(); // expected-error {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::array<int, 0> c0;
    c0.empty(); // expected-error {{ignoring return value of function declared with 'nodiscard' attribute}}

  return 0;
}
