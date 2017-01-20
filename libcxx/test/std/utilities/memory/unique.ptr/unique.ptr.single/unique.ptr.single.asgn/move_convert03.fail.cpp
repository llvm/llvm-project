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

// Test unique_ptr converting move assignment

// Can't assign from lvalue

#include <memory>

#include "test_macros.h"
#include "deleter_types.h"

struct A
{
    A() {}
    virtual ~A() {}
};

struct B : public A
{
};

// Can't assign from lvalue
int main()
{
    Deleter<B> db;
    std::unique_ptr<B, Deleter<B>& > s(new B, db);
    Deleter<A> da;
    std::unique_ptr<A, Deleter<A> &> s2(new A, da);
#if TEST_STD_VER >= 11
    s2 = s; // expected-error {{no viable overloaded '='}}
#else
    // NOTE: The move-semantic emulation creates an ambiguous overload set
    // so that assignment from an lvalue does not compile
    s2 = s; // expected-error {{use of overloaded operator '=' is ambiguous}}
#endif
}
