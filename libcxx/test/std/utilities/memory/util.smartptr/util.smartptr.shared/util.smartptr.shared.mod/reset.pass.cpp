//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// void reset();

#include <memory>
#include <cassert>

#include "test_macros.h"

struct B
{
    static int count;

    B() {++count;}
    B(const B&) {++count;}
    virtual ~B() {--count;}
};

int B::count = 0;

struct A
    : public B
{
    static int count;

    A() {++count;}
    A(const A& other) : B(other) {++count;}
    ~A() {--count;}
};

int A::count = 0;

int main(int, char**)
{
    {
        std::shared_ptr<B> p(new B);
        p.reset();
        assert(A::count == 0);
        assert(B::count == 0);
        assert(p.use_count() == 0);
        assert(p.get() == 0);
    }
    assert(A::count == 0);
    {
        std::shared_ptr<B> p;
        p.reset();
        assert(A::count == 0);
        assert(B::count == 0);
        assert(p.use_count() == 0);
        assert(p.get() == 0);
    }
    assert(A::count == 0);

  return 0;
}
