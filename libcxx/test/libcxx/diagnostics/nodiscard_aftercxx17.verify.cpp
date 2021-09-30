//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that _LIBCPP_NODISCARD_AFTER_CXX17 works
//	#define _LIBCPP_NODISCARD_AFTER_CXX17 [[nodiscard]]

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <__config>

_LIBCPP_NODISCARD_AFTER_CXX17 int foo() { return 6; }

int main(int, char**)
{
    foo(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    return 0;
}
