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

// class promise<R>

// promise(const promise&) = delete;

#include <future>

#include "test_macros.h"

int main(int, char**)
{
#if TEST_STD_VER >= 11
    {
        std::promise<int> p0;
        std::promise<int> p(p0); // expected-error {{call to deleted constructor of 'std::promise<int>'}}
    }
    {
        std::promise<int &> p0;
        std::promise<int &> p(p0); // expected-error {{call to deleted constructor of 'std::promise<int &>'}}
    }
    {
        std::promise<void> p0;
        std::promise<void> p(p0); // expected-error {{call to deleted constructor of 'std::promise<void>'}}
    }
#else
    {
        std::promise<int> p0;
        std::promise<int> p(p0); // expected-error {{calling a private constructor of class 'std::__1::promise<int>'}}
    }
    {
        std::promise<int &> p0;
        std::promise<int &> p(p0); // expected-error {{calling a private constructor of class 'std::__1::promise<int &>'}}
    }
    {
        std::promise<void> p0;
        std::promise<void> p(p0); // expected-error {{calling a private constructor of class 'std::__1::promise<void>'}}
    }
#endif

  return 0;
}
