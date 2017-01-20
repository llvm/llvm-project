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

// Test unique_ptr converting move ctor

#include <memory>
#include <utility>
#include <cassert>

#include "deleter_types.h"

// test converting move ctor.  Should only require a MoveConstructible deleter, or if
//    deleter is a reference, not even that.
// Implicit version

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

int main()
{
    std::unique_ptr<B, Deleter<B> > s(new B);
    std::unique_ptr<A, Deleter<A> > s2 = s;
}
