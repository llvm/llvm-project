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

int main()
{
    {
    std::unique_ptr<A[]> s(new A[3]);
    A* p = s.get();
    std::unique_ptr<A[]> s2 = std::move(s);
    assert(s2.get() == p);
    assert(s.get() == 0);
    assert(A::count == 3);
    }
    assert(A::count == 0);
    {
    std::unique_ptr<A[], Deleter<A[]> > s(new A[3], Deleter<A[]>(5));
    A* p = s.get();
    std::unique_ptr<A[], Deleter<A[]> > s2 = std::move(s);
    assert(s2.get() == p);
    assert(s.get() == 0);
    assert(A::count == 3);
    assert(s2.get_deleter().state() == 5);
    assert(s.get_deleter().state() == 0);
    }
    assert(A::count == 0);
    {
    NCDeleter d;
    std::unique_ptr<A[], NCDeleter&> s(new A[3], d);
    A* p = s.get();
    std::unique_ptr<A[], NCDeleter&> s2 = std::move(s);
    assert(s2.get() == p);
    assert(s.get() == 0);
    assert(A::count == 3);
    d.set_state(6);
    assert(s2.get_deleter().state() == d.state());
    assert(s.get_deleter().state() ==  d.state());
    }
    assert(A::count == 0);
}
