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

// class future<R>

// future(const future&) = delete;

#include <future>

void f() {
    {
        std::future<int> f0;
        std::future<int> f = f0; // expected-error {{call to deleted constructor of 'std::future<int>'}}
    }
    {
        std::future<int &> f0;
        std::future<int &> f = f0; // expected-error {{call to deleted constructor of 'std::future<int &>'}}
    }
    {
        std::future<void> f0;
        std::future<void> f = f0; // expected-error {{call to deleted constructor of 'std::future<void>'}}
    }
}
