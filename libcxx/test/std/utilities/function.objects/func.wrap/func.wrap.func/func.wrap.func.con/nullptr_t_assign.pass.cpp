//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R(ArgTypes...)>

// function& operator=(nullptr_t);

// This test runs in C++03, but we have deprecated using std::function in C++03.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <functional>
#include <cassert>

#include "count_new.h"

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

int g(int) {return 0;}

int main(int, char**)
{
    globalMemCounter.reset();
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
    std::function<int(int)> f = A();
    assert(A::count == 1);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    RTTI_ASSERT(f.target<A>());
    f = nullptr;
    assert(A::count == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    RTTI_ASSERT(f.target<A>() == 0);
    }
    {
    std::function<int(int)> f = g;
    assert(globalMemCounter.checkOutstandingNewEq(0));
    RTTI_ASSERT(f.target<int(*)(int)>());
    RTTI_ASSERT(f.target<A>() == 0);
    f = nullptr;
    assert(globalMemCounter.checkOutstandingNewEq(0));
    RTTI_ASSERT(f.target<int(*)(int)>() == 0);
    }

  return 0;
}
