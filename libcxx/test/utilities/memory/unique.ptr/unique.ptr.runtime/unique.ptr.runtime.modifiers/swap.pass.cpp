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

// test swap

#include <memory>
#include <cassert>

#include "../../deleter.h"

struct A
{
    int state_;
    static int count;
    A() : state_(0) {++count;}
    explicit A(int i) : state_(i) {++count;}
    A(const A& a) : state_(a.state_) {++count;}
    A& operator=(const A& a) {state_ = a.state_; return *this;}
    ~A() {--count;}

    friend bool operator==(const A& x, const A& y)
        {return x.state_ == y.state_;}
};

int A::count = 0;

int main()
{
    {
    A* p1 = new A[3];
    std::unique_ptr<A[], Deleter<A[]> > s1(p1, Deleter<A[]>(1));
    A* p2 = new A[3];
    std::unique_ptr<A[], Deleter<A[]> > s2(p2, Deleter<A[]>(2));
    assert(s1.get() == p1);
    assert(s1.get_deleter().state() == 1);
    assert(s2.get() == p2);
    assert(s2.get_deleter().state() == 2);
    s1.swap(s2);
    assert(s1.get() == p2);
    assert(s1.get_deleter().state() == 2);
    assert(s2.get() == p1);
    assert(s2.get_deleter().state() == 1);
    assert(A::count == 6);
    }
    assert(A::count == 0);
}
