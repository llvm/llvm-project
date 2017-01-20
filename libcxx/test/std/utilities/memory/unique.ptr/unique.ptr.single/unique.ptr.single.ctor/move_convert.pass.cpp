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

// NOTE: unique_ptr does not provide converting constructors in c++03
// XFAIL: c++98, c++03



#include <memory>
#include <type_traits>
#include <utility>
#include <cassert>

#include "deleter_types.h"

// test converting move ctor.  Should only require a MoveConstructible deleter, or if
//    deleter is a reference, not even that.
// Explicit version

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


template <class LHS, class RHS>
void checkReferenceDeleter(LHS& lhs, RHS& rhs) {
    typedef typename LHS::deleter_type NewDel;
    static_assert(std::is_reference<NewDel>::value, "");
    rhs.get_deleter().set_state(42);
    assert(rhs.get_deleter().state() == 42);
    assert(lhs.get_deleter().state() == 42);
    lhs.get_deleter().set_state(99);
    assert(lhs.get_deleter().state() == 99);
    assert(rhs.get_deleter().state() == 99);
}

template <class LHS, class RHS>
void checkDeleter(LHS& lhs, RHS& rhs, int LHSVal, int RHSVal) {
    assert(lhs.get_deleter().state() == LHSVal);
    assert(rhs.get_deleter().state() == RHSVal);
}

template <class LHS, class RHS>
void checkCtor(LHS& lhs, RHS& rhs, A* RHSVal) {
    assert(lhs.get() == RHSVal);
    assert(rhs.get() == nullptr);
    assert(A::count == 1);
    assert(B::count == 1);
}

void checkNoneAlive() {
    assert(A::count == 0);
    assert(B::count == 0);
}

int main()
{
    {
        typedef std::unique_ptr<A> APtr;
        typedef std::unique_ptr<B> BPtr;
        { // explicit
            BPtr b(new B);
            A* p = b.get();
            APtr a(std::move(b));
            checkCtor(a, b, p);
        }
        checkNoneAlive();
        { // implicit
            BPtr b(new B);
            A* p = b.get();
            APtr a = std::move(b);
            checkCtor(a, b, p);
        }
        checkNoneAlive();
    }
    { // test with moveable deleters
        typedef std::unique_ptr<A, Deleter<A> > APtr;
        typedef std::unique_ptr<B, Deleter<B> > BPtr;
        {
            Deleter<B> del(5);
            BPtr b(new B, std::move(del));
            A* p = b.get();
            APtr a(std::move(b));
            checkCtor(a, b, p);
            checkDeleter(a, b, 5, 0);
        }
        checkNoneAlive();
        {
            Deleter<B> del(5);
            BPtr b(new B, std::move(del));
            A* p = b.get();
            APtr a = std::move(b);
            checkCtor(a, b, p);
            checkDeleter(a, b, 5, 0);
        }
        checkNoneAlive();

    }
    { // test with reference deleters
        typedef std::unique_ptr<A, NCDeleter<A>& > APtr;
        typedef std::unique_ptr<B, NCDeleter<A>& > BPtr;
        NCDeleter<A> del(5);
        {
            BPtr b(new B, del);
            A* p = b.get();
            APtr a(std::move(b));
            checkCtor(a, b, p);
            checkReferenceDeleter(a, b);
        }
        checkNoneAlive();
        {
            BPtr b(new B, del);
            A* p = b.get();
            APtr a = std::move(b);
            checkCtor(a, b, p);
            checkReferenceDeleter(a, b);
        }
        checkNoneAlive();
    }
    {
        typedef std::unique_ptr<A, CDeleter<A> > APtr;
        typedef std::unique_ptr<B, CDeleter<B>& > BPtr;
        CDeleter<B> del(5);
        {
            BPtr b(new B, del);
            A* p = b.get();
            APtr a(std::move(b));
            checkCtor(a, b, p);
            checkDeleter(a, b, 5, 5);
        }
        checkNoneAlive();
        {
            BPtr b(new B, del);
            A* p = b.get();
            APtr a = std::move(b);
            checkCtor(a, b, p);
            checkDeleter(a, b, 5, 5);
        }
        checkNoneAlive();
    }
}
