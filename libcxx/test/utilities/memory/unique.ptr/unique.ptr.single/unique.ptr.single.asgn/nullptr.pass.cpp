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

// Test unique_ptr move assignment

#include <memory>
#include <cassert>

// test assignment from null

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
    std::unique_ptr<A> s2(new A);
    assert(A::count == 1);
    s2 = nullptr;
    assert(A::count == 0);
    assert(s2.get() == 0);
    }
    assert(A::count == 0);
}
