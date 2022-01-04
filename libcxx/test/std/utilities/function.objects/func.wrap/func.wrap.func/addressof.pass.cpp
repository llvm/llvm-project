//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R(ArgTypes...)>

// This test runs in C++03, but we have deprecated using std::function in C++03.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// Make sure we can use std::function with a type that has a hostile overload
// of operator&().

#include <functional>
#include <cassert>

#include "operator_hijacker.h"

struct TrapAddressof : operator_hijacker {
    int operator()() const { return 1; }
};

int main(int, char**) {
    std::function<int()> f = TrapAddressof();
    assert(f() == 1);
    return 0;
}
