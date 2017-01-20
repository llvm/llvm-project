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
    const std::unique_ptr<B> s(new B);
    std::unique_ptr<A> s2;
#if TEST_STD_VER >= 11
    s2 = s; // expected-error {{no viable overloaded '='}}
#else
    // NOTE: The error says "constructor" because the assignment operator takes
    // 's' by value and attempts to copy construct it.
    s2 = s; // expected-error {{no matching constructor for initialization}}
#endif
}
