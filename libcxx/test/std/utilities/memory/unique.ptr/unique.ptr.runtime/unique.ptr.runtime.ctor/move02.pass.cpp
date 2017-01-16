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


std::unique_ptr<A[]>
source1()
{
    return std::unique_ptr<A[]>(new A[3]);
}

void sink1(std::unique_ptr<A[]>)
{
}

std::unique_ptr<A[], Deleter<A[]> >
source2()
{
    return std::unique_ptr<A[], Deleter<A[]> >(new A[3]);
}

void sink2(std::unique_ptr<A[], Deleter<A[]> >)
{
}

std::unique_ptr<A[], NCDeleter<A[]>&>
source3()
{
    static NCDeleter<A[]> d;
    return std::unique_ptr<A[], NCDeleter<A[]>&>(new A[3], d);
}

void sink3(std::unique_ptr<A[], NCDeleter<A[]>&>)
{
}

int main()
{
    sink1(source1());
    sink2(source2());
    sink3(source3());
    assert(A::count == 0);
}
