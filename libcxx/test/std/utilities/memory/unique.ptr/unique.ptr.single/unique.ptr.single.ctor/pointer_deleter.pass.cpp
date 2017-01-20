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

//=============================================================================
// TESTING unique_ptr(pointer, deleter)
//
// Concerns:
//   1 unique_ptr(pointer, deleter&&) only requires a MoveConstructible deleter.
//   2 unique_ptr(pointer, deleter&) requires a CopyConstructible deleter.
//   3 unique_ptr<T, D&>(pointer, deleter) does not require a CopyConstructible deleter.
//   4 unique_ptr<T, D const&>(pointer, deleter) does not require a CopyConstructible deleter.
//   5 unique_ptr(pointer, deleter) should work for derived pointers.
//   6 unique_ptr(pointer, deleter) should work with function pointers.
//   7 unique_ptr<void> should work.


#include <memory>
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

bool my_free_called = false;

void my_free(void*) {
    my_free_called = true;
}

int main()
{
    { // MoveConstructible deleter (C-1)
        A* p = new A;
        assert(A::count == 1);
        std::unique_ptr<A, Deleter<A> > s(p, Deleter<A>(5));
        assert(s.get() == p);
        assert(s.get_deleter().state() == 5);
    }
    assert(A::count == 0);
    { // CopyConstructible deleter (C-2)
        A* p = new A;
        assert(A::count == 1);
        CopyDeleter<A> d(5);
        std::unique_ptr<A, CopyDeleter<A> > s(p, d);
        assert(s.get() == p);
        assert(s.get_deleter().state() == 5);
        d.set_state(6);
        assert(s.get_deleter().state() == 5);
    }
    assert(A::count == 0);
    { // Reference deleter (C-3)
        A* p = new A;
        assert(A::count == 1);
        NCDeleter<A> d(5);
        std::unique_ptr<A, NCDeleter<A>&> s(p, d);
        assert(s.get() == p);
        assert(&s.get_deleter() == &d);
        assert(s.get_deleter().state() == 5);
        d.set_state(6);
        assert(s.get_deleter().state() == 6);
    }
    assert(A::count == 0);
    { // Const Reference deleter (C-4)
        A* p = new A;
        assert(A::count == 1);
        NCConstDeleter<A> d(5);
        std::unique_ptr<A, NCConstDeleter<A> const&> s(p, d);
        assert(s.get() == p);
        assert(s.get_deleter().state() == 5);
        assert(&s.get_deleter() == &d);
    }
    assert(A::count == 0);
    { // Derived pointers (C-5)
        B* p = new B;
        assert(A::count == 1);
        assert(B::count == 1);
        std::unique_ptr<A, Deleter<A> > s(p, Deleter<A>(5));
        assert(s.get() == p);
        assert(s.get_deleter().state() == 5);
    }
    assert(A::count == 0);
    assert(B::count == 0);
    { // Void and function pointers (C-6,7)
        {
        int i = 0;
        std::unique_ptr<void, void(*)(void*)> s(&i, my_free);
        assert(s.get() == &i);
        assert(s.get_deleter() == my_free);
        assert(!my_free_called);
        }
        assert(my_free_called);
    }
}
