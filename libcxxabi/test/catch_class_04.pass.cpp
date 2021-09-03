//===---------------------- catch_class_04.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
    This test checks that adjustedPtr is correct as there exist offsets in this
    object for the various subobjects, all of which have a unique id_ to
    check against.  It also checks that virtual bases work properly
*/

// UNSUPPORTED: no-exceptions

// Compilers emit warnings about exceptions of type 'Child' being caught by
// an earlier handler of type 'Base'. Congrats, you've just diagnosed the
// behavior under test.
// ADDITIONAL_COMPILE_FLAGS: -Wno-exceptions

#include <exception>
#include <stdlib.h>
#include <assert.h>

struct B
{
    static int count;
    int id_;
    explicit B(int id) : id_(id) {count++;}
    B(const B& a) : id_(a.id_) {count++;}
    ~B() {count--;}
};

int B::count = 0;

struct C1
    : virtual B
{
    static int count;
    int id_;
    explicit C1(int id) : B(id-2), id_(id) {count++;}
    C1(const C1& a) : B(a.id_-2), id_(a.id_) {count++;}
    ~C1() {count--;}
};

int C1::count = 0;

struct C2
    : virtual private B
{
    static int count;
    int id_;
    explicit C2(int id) : B(id-2), id_(id) {count++;}
    C2(const C2& a) : B(a.id_-2), id_(a.id_) {count++;}
    ~C2() {count--;}
};

int C2::count = 0;

struct A
    : C1, C2
{
    static int count;
    int id_;
    explicit A(int id) : B(id+3), C1(id-1), C2(id-2), id_(id) {count++;}
    A(const A& a) :  B(a.id_+3), C1(a.id_-1), C2(a.id_-2), id_(a.id_) {count++;}
    ~A() {count--;}
};

int A::count = 0;

void f1()
{
    assert(A::count == 0);
    assert(C1::count == 0);
    assert(C2::count == 0);
    assert(B::count == 0);
    A a(5);
    assert(A::count == 1);
    assert(C1::count == 1);
    assert(C2::count == 1);
    assert(B::count == 1);

    assert(a.id_ == 5);
    assert(static_cast<C1&>(a).id_ == 4);
    assert(static_cast<C2&>(a).id_ == 3);
    assert(static_cast<B&>(a).id_ == 8);
    throw a;
    assert(false);
}

void f2()
{
    try
    {
        assert(A::count == 0);
        assert(C1::count == 0);
        assert(C2::count == 0);
        assert(B::count == 0);
        f1();
        assert(false);
    }
    catch (const A& a)  // can catch A
    {
        assert(a.id_ == 5);
        assert(static_cast<const C1&>(a).id_ == 4);
        assert(static_cast<const C2&>(a).id_ == 3);
        assert(static_cast<const B&>(a).id_ == 8);
        throw;
    }
    catch (const C1&)
    {
        assert(false);
    }
    catch (const C2&)
    {
        assert(false);
    }
    catch (const B&)
    {
        assert(false);
    }
}

void f3()
{
    try
    {
        assert(A::count == 0);
        assert(C1::count == 0);
        assert(C2::count == 0);
        assert(B::count == 0);
        f2();
        assert(false);
    }
    catch (const B& a)  // can catch B
    {
        assert(static_cast<const B&>(a).id_ == 8);
        throw;
    }
    catch (const C1& c1)
    {
        assert(false);
    }
    catch (const C2&)
    {
        assert(false);
    }
}

void f4()
{
    try
    {
        assert(A::count == 0);
        assert(C1::count == 0);
        assert(C2::count == 0);
        assert(B::count == 0);
        f3();
        assert(false);
    }
    catch (const C2& c2)  // can catch C2
    {
        assert(c2.id_ == 3);
        throw;
    }
    catch (const B& a)  // can not catch B (ambiguous base)
    {
        assert(false);
    }
    catch (const C1&)
    {
        assert(false);
    }
}

void f5()
{
    try
    {
        assert(A::count == 0);
        assert(C1::count == 0);
        assert(C2::count == 0);
        assert(B::count == 0);
        f4();
        assert(false);
    }
    catch (const C1& c1)  // can catch C1
    {
        assert(c1.id_ == 4);
        assert(static_cast<const B&>(c1).id_ == 8);
        throw;
    }
    catch (const B& a)
    {
        assert(false);
    }
    catch (const C2&)
    {
        assert(false);
    }
}

int main(int, char**)
{
    try
    {
        f5();
        assert(false);
    }
    catch (...)
    {
    }
    assert(A::count == 0);
    assert(C1::count == 0);
    assert(C2::count == 0);
    assert(B::count == 0);

    return 0;
}
