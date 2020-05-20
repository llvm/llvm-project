//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
// XFAIL: c++98, c++03, c++11, c++14

// class function<R(ArgTypes...)>

// template<class A> function(allocator_arg_t, const A&, function&&);
//
// This signature was removed in C++17

// This test runs in C++03, but we have deprecated using std::function in C++03.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <memory>
#include <cassert>

#include "test_macros.h"

class A
{
    int data_[10];
public:
    static int count;

    A()
    {
        ++count;
        for (int i = 0; i < 10; ++i)
            data_[i] = i;
    }

    A(const A&) {++count;}

    ~A() {--count;}

    int operator()(int i) const
    {
        for (int j = 0; j < 10; ++j)
            i += data_[j];
        return i;
    }
};

int A::count = 0;

int g(int) { return 0; }

int main(int, char**)
{
    {
        std::function<int(int)> f = A();
        std::function<int(int)> f2(std::allocator_arg, std::allocator<A>(), std::move(f));
    }

  return 0;
}
