//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:
// pointer allocate(size_type n, allocator<void>::const_pointer hint=0);

#include <memory>
#include <cassert>

#include "count_new.hpp"

int A_constructed = 0;

struct A
{
    int data;
    A() {++A_constructed;}
    A(const A&) {++A_constructed;}
    ~A() {--A_constructed;}
};

int main()
{
    globalMemCounter.reset();
    std::allocator<A> a;
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(A_constructed == 0);
    globalMemCounter.last_new_size = 0;
    A* volatile ap = a.allocate(3);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(globalMemCounter.checkLastNewSizeEq(3 * sizeof(int)));
    assert(A_constructed == 0);
    a.deallocate(ap, 3);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(A_constructed == 0);

    globalMemCounter.last_new_size = 0;
    A* volatile ap2 = a.allocate(3, (const void*)5);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(globalMemCounter.checkLastNewSizeEq(3 * sizeof(int)));
    assert(A_constructed == 0);
    a.deallocate(ap2, 3);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(A_constructed == 0);
}
