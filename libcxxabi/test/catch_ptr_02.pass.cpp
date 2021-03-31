//===------------------------- catch_ptr_02.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// FIXME: GCC doesn't allow turning off the warning for exceptions being caught
//        by earlier handlers, which this test is exercising. We have to disable
//        warnings altogether to remove the error.
//        See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=97675.
// ADDITIONAL_COMPILE_FLAGS: -Wno-error

// The fix for PR17222 made it in the dylib for macOS 10.10
// XFAIL: with_system_cxx_lib=macosx10.9

#include <cassert>

// Clang emits  warnings about exceptions of type 'Child' being caught by
// an earlier handler of type 'Base'. Congrats clang, you've just
// diagnosed the behavior under test.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wexceptions"
#endif

#if __cplusplus < 201103L
#define DISABLE_NULLPTR_TESTS
#endif

struct  A {};
A a;
const A ca = A();

void test1 ()
{
    try
    {
        throw &a;
        assert(false);
    }
    catch ( const A* )
    {
    }
    catch ( A *)
    {
        assert (false);
    }
}

void test2 ()
{
    try
     {
        throw &a;
        assert(false);
    }
    catch ( A* )
    {
    }
    catch ( const A *)
    {
         assert (false);
    }
}

void test3 ()
{
    try
    {
        throw &ca;
        assert(false);
    }
    catch ( const A* )
    {
    }
    catch ( A *)
    {
        assert (false);
    }
}

void test4 ()
{
    try
    {
        throw &ca;
        assert(false);
    }
    catch ( A *)
    {
        assert (false);
    }
    catch ( const A* )
    {
    }
}

struct base1 {int x;};
struct base2 {int x;};
struct derived : base1, base2 {};

void test5 ()
{
    try
    {
        throw (derived*)0;
        assert(false);
    }
    catch (base2 *p) {
        assert (p == 0);
    }
    catch (...)
    {
        assert (false);
    }
}

void test6 ()
{
#if !defined(DISABLE_NULLPTR_TESTS)
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (base2 *p) {
        assert (p == nullptr);
    }
    catch (...)
    {
        assert (false);
    }
#endif
}

void test7 ()
{
    try
    {
        throw (derived*)12;
        assert(false);
    }
    catch (base2 *p) {
        assert ((unsigned long)p == 12+sizeof(base1));
    }
    catch (...)
    {
        assert (false);
    }
}


struct vBase {};
struct vDerived : virtual public vBase {};

void test8 ()
{
    vDerived derived;
    try
    {
        throw &derived;
        assert(false);
    }
    catch (vBase *p) {
        assert(p != 0);
    }
    catch (...)
    {
        assert (false);
    }
}

void test9 ()
{
#if !defined(DISABLE_NULLPTR_TESTS)
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (vBase *p) {
        assert(p == 0);
    }
    catch (...)
    {
        assert (false);
    }
#endif
}

void test10 ()
{
    try
    {
        throw (vDerived*)0;
        assert(false);
    }
    catch (vBase *p) {
        assert(p == 0);
    }
    catch (...)
    {
        assert (false);
    }
}

int main(int, char**)
{
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    test8();
    test9();
    test10();

    return 0;
}
