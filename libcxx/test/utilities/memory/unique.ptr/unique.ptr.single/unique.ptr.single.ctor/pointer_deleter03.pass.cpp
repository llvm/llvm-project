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

// Test unique_ptr(pointer) ctor

#include <memory>
#include <cassert>

// unique_ptr<T, D&>(pointer, d) does not requires CopyConstructible deleter

struct A
{
    static int count;
    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

class Deleter
{
    int state_;

    Deleter(const Deleter&);
    Deleter& operator=(const Deleter&);
public:

    Deleter() : state_(5) {}

    int state() const {return state_;}
    void set_state(int s) {state_ = s;}

    void operator()(A* p) {delete p;}
};

int main()
{
    {
    A* p = new A;
    assert(A::count == 1);
    Deleter d;
    std::unique_ptr<A, Deleter&> s(p, d);
    assert(s.get() == p);
    assert(s.get_deleter().state() == 5);
    d.set_state(6);
    assert(s.get_deleter().state() == 6);
    }
    assert(A::count == 0);
}
