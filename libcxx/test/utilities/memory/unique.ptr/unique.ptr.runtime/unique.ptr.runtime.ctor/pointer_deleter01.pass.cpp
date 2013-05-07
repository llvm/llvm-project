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

// Test unique_ptr(pointer, deleter) ctor

// unique_ptr(pointer, deleter()) only requires MoveConstructible deleter

#include <memory>
#include <cassert>

#include "../../deleter.h"

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
    A* p = new A[3];
    assert(A::count == 3);
    std::unique_ptr<A[], Deleter<A[]> > s(p, Deleter<A[]>());
    assert(s.get() == p);
    assert(s.get_deleter().state() == 0);
    }
    assert(A::count == 0);
}
