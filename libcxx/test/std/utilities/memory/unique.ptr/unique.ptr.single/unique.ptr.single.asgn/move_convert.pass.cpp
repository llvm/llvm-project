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

// Test unique_ptr converting move assignment

#include <memory>
#include <utility>
#include <cassert>

#include "deleter_types.h"

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


template <class APtr, class BPtr>
void testAssign(APtr& aptr, BPtr& bptr) {
    A* p = bptr.get();
    assert(A::count == 2);
    aptr = std::move(bptr);
    assert(aptr.get() == p);
    assert(bptr.get() == 0);
    assert(A::count == 1);
    assert(B::count == 1);
}

template <class LHS, class RHS>
void checkDeleter(LHS& lhs, RHS& rhs, int LHSState, int RHSState) {
    assert(lhs.get_deleter().state() == LHSState);
    assert(rhs.get_deleter().state() == RHSState);
}

int main()
{
    {
        std::unique_ptr<B> bptr(new B);
        std::unique_ptr<A> aptr(new A);
        testAssign(aptr, bptr);
    }
    assert(A::count == 0);
    assert(B::count == 0);
    {
        Deleter<B> del(42);
        std::unique_ptr<B, Deleter<B> > bptr(new B, std::move(del));
        std::unique_ptr<A, Deleter<A> > aptr(new A);
        testAssign(aptr, bptr);
        checkDeleter(aptr, bptr, 42, 0);
    }
    assert(A::count == 0);
    assert(B::count == 0);
    {
        CDeleter<A> adel(6);
        CDeleter<B> bdel(42);
        std::unique_ptr<B, CDeleter<B>&> bptr(new B, bdel);
        std::unique_ptr<A, CDeleter<A>&> aptr(new A, adel);
        testAssign(aptr, bptr);
        checkDeleter(aptr, bptr, 42, 42);
    }
    assert(A::count == 0);
    assert(B::count == 0);
}
