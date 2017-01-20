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

#include <memory>
#include <utility>
#include <cassert>

#include "deleter_types.h"

//=============================================================================
// TESTING unique_ptr(unique_ptr&&)
//
// Concerns
//   1 The moved from pointer is empty and the new pointer stores the old value.
//   2 The only requirement on the deleter is that it is MoveConstructible
//     or a reference.
//   3 The constructor works for explicitly moved values (ie std::move(x))
//   4 The constructor works for true temporaries (ie a return value)
//
// Plan
//  1 Explicitly construct unique_ptr<T, D> for various deleter types 'D'.
//    check that the value and deleter have been properly moved. (C-1,2,3)
//
//  2 Use the expression 'sink(source())' to move construct a unique_ptr<T, D>
//    from a temporary. 'source' should return the unique_ptr by value and
//    'sink' should accept the unique_ptr by value. (C-1,2,4)

struct A
{
    static int count;
    A() {++count;}
    A(const A&) {++count;}
    virtual ~A() {--count;}
};

int A::count = 0;

template <class Expect>
void sinkFunction(Expect)
{
}

typedef std::unique_ptr<A> APtrSource1;
typedef std::unique_ptr<A, Deleter<A> > APtrSource2;
typedef std::unique_ptr<A, NCDeleter<A>& > APtrSource3;

APtrSource1 source1() {
    return APtrSource1 (new A);
}

void sink1(APtrSource1 p) {
    assert(p.get() != nullptr);
}

APtrSource2 source2() {
    return APtrSource2(new A, Deleter<A>(5));
}

void sink2(APtrSource2 p) {
    assert(p.get() != nullptr);
    assert(p.get_deleter().state() == 5);
}

APtrSource3 source3() {
    static NCDeleter<A> d(5);
    return APtrSource3(new A, d);
}

void sink3(APtrSource3 p) {
    assert(p.get() != nullptr);
    assert(p.get_deleter().state() == 5);
    assert(&p.get_deleter() == &source3().get_deleter());
}

int main()
{
    {
        typedef std::unique_ptr<A> APtr;
        APtr s(new A);
        A* p = s.get();
        APtr s2 = std::move(s);
        assert(s2.get() == p);
        assert(s.get() == 0);
        assert(A::count == 1);
    }
    assert(A::count == 0);
    {
        typedef Deleter<A> MoveDel;
        typedef std::unique_ptr<A, MoveDel> APtr;
        MoveDel d(5);
        APtr s(new A, std::move(d));
        assert(d.state() == 0);
        assert(s.get_deleter().state() == 5);
        A* p = s.get();
        APtr s2 = std::move(s);
        assert(s2.get() == p);
        assert(s.get() == 0);
        assert(A::count == 1);
        assert(s2.get_deleter().state() == 5);
        assert(s.get_deleter().state() == 0);
    }
    assert(A::count == 0);
    {
        typedef NCDeleter<A> NonCopyDel;
        typedef std::unique_ptr<A, NonCopyDel&> APtr;

        NonCopyDel d;
        APtr s(new A, d);
        A* p = s.get();
        APtr s2 = std::move(s);
        assert(s2.get() == p);
        assert(s.get() == 0);
        assert(A::count == 1);
        d.set_state(6);
        assert(s2.get_deleter().state() == d.state());
        assert(s.get_deleter().state() ==  d.state());
    }
    assert(A::count == 0);
    {
       sink1(source1());
       assert(A::count == 0);
       sink2(source2());
       assert(A::count == 0);
       sink3(source3());
       assert(A::count == 0);
    }
    assert(A::count == 0);
}
