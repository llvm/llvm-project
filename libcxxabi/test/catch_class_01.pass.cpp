//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

#include <exception>
#include <stdlib.h>
#include <assert.h>

struct A
{
    static int count;
    int id_;
    explicit A(int id) : id_(id) {count++;}
    A(const A& a) : id_(a.id_) {count++;}
    ~A() {count--;}
};

int A::count = 0;

void f1()
{
    throw A(3);
}

void f2()
{
    try
    {
        assert(A::count == 0);
        f1();
    }
    catch (A a)
    {
        assert(A::count != 0);
        assert(a.id_ == 3);
        throw;
    }
}

int main(int, char**)
{
    try
    {
        f2();
        assert(false);
    }
    catch (const A& a)
    {
        assert(A::count != 0);
        assert(a.id_ == 3);
    }
    assert(A::count == 0);

    return 0;
}
