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

// test move assignment.  Should only require a MoveConstructible deleter, or if
//    deleter is a reference, not even that.

#include <memory>
#include <cassert>

#include "../deleter.h"

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
    std::unique_ptr<A[]> s1(new A[3]);
    A* p = s1.get();
    assert(A::count == 3);
    std::unique_ptr<A[]> s2(new A[2]);
    assert(A::count == 5);
    s2 = std::move(s1);
    assert(A::count == 3);
    assert(s2.get() == p);
    assert(s1.get() == 0);
    }
    assert(A::count == 0);
    {
    std::unique_ptr<A[], Deleter<A[]> > s1(new A[4], Deleter<A[]>(5));
    A* p = s1.get();
    assert(A::count == 4);
    std::unique_ptr<A[], Deleter<A[]> > s2(new A[5]);
    assert(A::count == 9);
    s2 = std::move(s1);
    assert(s2.get() == p);
    assert(s1.get() == 0);
    assert(A::count == 4);
    assert(s2.get_deleter().state() == 5);
    assert(s1.get_deleter().state() == 0);
    }
    assert(A::count == 0);
    {
    CDeleter<A[]> d1(5);
    std::unique_ptr<A[], CDeleter<A[]>&> s1(new A[6], d1);
    A* p = s1.get();
    assert(A::count == 6);
    CDeleter<A[]> d2(6);
    std::unique_ptr<A[], CDeleter<A[]>&> s2(new A[3], d2);
    assert(A::count == 9);
    s2 = std::move(s1);
    assert(A::count == 6);
    assert(s2.get() == p);
    assert(s1.get() == 0);
    assert(d1.state() == 5);
    assert(d2.state() == 5);
    }
    assert(A::count == 0);
}
