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

// Test unique_ptr<T[]>(pointer) ctor

// unique_ptr<T[]>(pointer) ctor shouldn't require complete type

#include <memory>
#include <cassert>

struct A;

class Deleter
{
    int state_;

    Deleter(Deleter&);
    Deleter& operator=(Deleter&);

public:
    Deleter() : state_(5) {}

    int state() const {return state_;}

    void operator()(A* p);
};

void check(int i);

template <class D = std::default_delete<A[]> >
struct B
{
    std::unique_ptr<A[], D> a_;
    explicit B(A*);
    ~B();

    A* get() const {return a_.get();}
    D& get_deleter() {return a_.get_deleter();}
};

A* get();

int main()
{
    {
    A* p = get();
    check(3);
    B<> s(p);
    assert(s.get() == p);
    }
    check(0);
    {
    A* p = get();
    check(3);
    B<Deleter> s(p);
    assert(s.get() == p);
    assert(s.get_deleter().state() == 5);
    }
    check(0);
}

struct A
{
    static int count;
    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

A* get() {return new A[3];}

void Deleter::operator()(A* p) {delete [] p;}

void check(int i)
{
    assert(A::count == i);
}

template <class D>
B<D>::B(A* a) : a_(a) {}

template <class D>
B<D>::~B() {}
