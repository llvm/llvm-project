//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// default_delete

#include <memory>
#include <cassert>

struct A
{
    static int count;
    A() {++count;}
    A(const A&) {++count;}
    virtual ~A() {--count;}
};

int A::count = 0;

struct B
    : public A
{
    static int count;
    B() {++count;}
    B(const B&) {++count;}
    virtual ~B() {--count;}
};

int B::count = 0;

int main(int, char**)
{
    std::default_delete<B> d2;
    std::default_delete<A> d1 = d2;
    A* p = new B;
    assert(A::count == 1);
    assert(B::count == 1);
    d1(p);
    assert(A::count == 0);
    assert(B::count == 0);

  return 0;
}
