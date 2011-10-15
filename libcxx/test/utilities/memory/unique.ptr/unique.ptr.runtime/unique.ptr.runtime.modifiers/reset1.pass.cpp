//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test reset

#include <memory>
#include <cassert>

struct A
{
    static int count;
    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

int main()
{
    {
    std::unique_ptr<A[]> p(new A[3]);
    assert(A::count == 3);
    A* i = p.get();
    p.reset();
    assert(A::count == 0);
    assert(p.get() == 0);
    }
    assert(A::count == 0);
    {
    std::unique_ptr<A[]> p(new A[4]);
    assert(A::count == 4);
    A* i = p.get();
    p.reset(new A[5]);
    assert(A::count == 5);
    }
    assert(A::count == 0);
}
