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
// TESTING std::unique_ptr::unique_ptr()
//
// Concerns:
//   1 The pointer constructor works for any default constructible deleter types.
//   2 The pointer constructor accepts pointers to derived types.
//   2 The stored type 'T' is allowed to be incomplete.
//
// Plan
//  1 Construct unique_ptr<T, D>'s with a pointer to 'T' and various deleter
//   types (C-1)
//  2 Construct unique_ptr<T, D>'s with a pointer to 'D' and various deleter
//    types where 'D' is derived from 'T'. (C-1,2)
//  3 Construct a unique_ptr<T, D> with a pointer to 'T' and various deleter
//    types where 'T' is an incomplete type (C-1,3)

// Test unique_ptr(pointer) ctor

#include <memory>
#include <cassert>

#include "deleter_types.h"

// unique_ptr(pointer) ctor should only require default Deleter ctor

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


struct IncompleteT;

IncompleteT* getIncomplete();
void checkNumIncompleteTypeAlive(int i);

template <class Del = std::default_delete<IncompleteT> >
struct StoresIncomplete {
  std::unique_ptr<IncompleteT, Del> m_ptr;
  StoresIncomplete() {}
  explicit StoresIncomplete(IncompleteT* ptr) : m_ptr(ptr) {}
  ~StoresIncomplete();

  IncompleteT* get() const { return m_ptr.get(); }
  Del& get_deleter() { return m_ptr.get_deleter(); }
};

void test_pointer()
{
    {
        A* p = new A;
        assert(A::count == 1);
        std::unique_ptr<A> s(p);
        assert(s.get() == p);
    }
    assert(A::count == 0);
    {
        A* p = new A;
        assert(A::count == 1);
        std::unique_ptr<A, NCDeleter<A> > s(p);
        assert(s.get() == p);
        assert(s.get_deleter().state() == 0);
    }
    assert(A::count == 0);
}

void test_derived()
{
    {
        B* p = new B;
        assert(A::count == 1);
        assert(B::count == 1);
        std::unique_ptr<A> s(p);
        assert(s.get() == p);
    }
    assert(A::count == 0);
    assert(B::count == 0);
    {
        B* p = new B;
        assert(A::count == 1);
        assert(B::count == 1);
        std::unique_ptr<A, NCDeleter<A> > s(p);
        assert(s.get() == p);
        assert(s.get_deleter().state() == 0);
    }
    assert(A::count == 0);
    assert(B::count == 0);
}

void test_incomplete()
{
    {
        IncompleteT* p = getIncomplete();
        checkNumIncompleteTypeAlive(1);
        StoresIncomplete<> s(p);
        assert(s.get() == p);
    }
    checkNumIncompleteTypeAlive(0);
    {
        IncompleteT* p = getIncomplete();
        checkNumIncompleteTypeAlive(1);
        StoresIncomplete< NCDeleter<IncompleteT> > s(p);
        assert(s.get() == p);
        assert(s.get_deleter().state() == 0);
    }
    checkNumIncompleteTypeAlive(0);
}

struct IncompleteT {
    static int count;
    IncompleteT() { ++count; }
    ~IncompleteT() {--count; }
};

int IncompleteT::count = 0;

IncompleteT* getIncomplete() {
    return new IncompleteT;
}

void checkNumIncompleteTypeAlive(int i) {
    assert(IncompleteT::count == i);
}

template <class Del>
StoresIncomplete<Del>::~StoresIncomplete() { }

int main()
{
    test_pointer();
    test_derived();
    test_incomplete();
}
