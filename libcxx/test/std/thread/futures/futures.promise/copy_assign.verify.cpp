//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <future>

// class promise<R>

// promise& operator=(const promise& rhs) = delete;

#include <future>

void f() {
    {
        std::promise<int> p0, p;
        p = p0; // expected-error {{overload resolution selected deleted operator '='}}
    }
    {
        std::promise<int&> p0, p;
        p = p0; // expected-error {{overload resolution selected deleted operator '='}}
    }
    {
        std::promise<void> p0, p;
        p = p0; // expected-error {{overload resolution selected deleted operator '='}}
    }
}
