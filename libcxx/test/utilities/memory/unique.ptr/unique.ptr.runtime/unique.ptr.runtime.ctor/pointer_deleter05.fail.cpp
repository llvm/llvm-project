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

// unique_ptr(pointer, deleter) should not work with derived pointers

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

class Deleter
{
    int state_;

public:
    Deleter() : state_(5) {}

    int state() const {return state_;}

    void operator()(A* p) {delete [] p;}
};

int main()
{
    B* p = new B[3];
    std::unique_ptr<A[], Deleter> s(p, Deleter());
}
