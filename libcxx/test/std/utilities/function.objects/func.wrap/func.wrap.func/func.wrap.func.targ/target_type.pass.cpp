//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// class function<R(ArgTypes...)>

// const std::type_info& target_type() const;

// UNSUPPORTED: no-rtti

#include <functional>
#include <typeinfo>
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

    int foo(int) const {return 1;}
};

int A::count = 0;

int g(int) {return 0;}

int main(int, char**)
{
    {
    std::function<int(int)> f = A();
    assert(f.target_type() == typeid(A));
    }
    {
    std::function<int(int)> f;
    assert(f.target_type() == typeid(void));
    }

  return 0;
}
