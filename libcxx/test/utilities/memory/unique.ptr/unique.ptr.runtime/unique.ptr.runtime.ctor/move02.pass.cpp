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

// Test unique_ptr move ctor

// test move ctor.  Should only require a MoveConstructible deleter, or if
//    deleter is a reference, not even that.

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

class NCDeleter
{
    int state_;

    NCDeleter(NCDeleter&);
    NCDeleter& operator=(NCDeleter&);
public:

    NCDeleter() : state_(5) {}

    int state() const {return state_;}
    void set_state(int s) {state_ = s;}

    void operator()(A* p) {delete [] p;}
};

std::unique_ptr<A[]>
source1()
{
    return std::unique_ptr<A[]>(new A[3]);
}

void sink1(std::unique_ptr<A[]> p)
{
}

std::unique_ptr<A[], Deleter<A[]> >
source2()
{
    return std::unique_ptr<A[], Deleter<A[]> >(new A[3]);
}

void sink2(std::unique_ptr<A[], Deleter<A[]> > p)
{
}

std::unique_ptr<A[], NCDeleter&>
source3()
{
    static NCDeleter d;
    return std::unique_ptr<A[], NCDeleter&>(new A[3], d);
}

void sink3(std::unique_ptr<A[], NCDeleter&> p)
{
}

int main()
{
    sink1(source1());
    sink2(source2());
    sink3(source3());
    assert(A::count == 0);
}
