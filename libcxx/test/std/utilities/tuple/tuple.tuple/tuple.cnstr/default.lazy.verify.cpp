//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test makes sure that we don't evaluate `is_default_constructible<T>`
// too early in std::tuple's default constructor.

// UNSUPPORTED: c++03

#include <tuple>

#include "test_macros.h"

struct Outer {
    template <class T>
    struct Inner {
        bool foo = false;
    };
    std::tuple<Inner<int>> tup;
};

Outer x; // expected-no-diagnostics
