//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <future>

// class future<R>

// future& operator=(const future&) = delete;

#include <future>

#include "test_macros.h"

int main(int, char**)
{
#if TEST_STD_VER >= 11
    {
        std::future<int> f0, f;
        f = f0; // expected-error {{overload resolution selected deleted operator '='}}
    }
    {
        std::future<int &> f0, f;
        f = f0; // expected-error {{overload resolution selected deleted operator '='}}
    }
    {
        std::future<void> f0, f;
        f = f0; // expected-error {{overload resolution selected deleted operator '='}}
    }
#else
    {
        std::future<int> f0, f;
        f = f0; // expected-error {{'operator=' is a private member of 'std::__1::future<int>'}}
    }
    {
        std::future<int &> f0, f;
        f = f0; // expected-error {{'operator=' is a private member of 'std::__1::future<int &>'}}
    }
    {
        std::future<void> f0, f;
        f = f0; // expected-error {{'operator=' is a private member of 'std::__1::future<void>'}}
    }
#endif

  return 0;
}
