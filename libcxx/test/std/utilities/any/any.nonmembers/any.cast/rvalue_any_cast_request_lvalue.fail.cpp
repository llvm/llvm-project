//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <any>

// template <class ValueType>
// ValueType any_cast(any &&);

// Try and use the rvalue any_cast to cast to an lvalue reference

#include <any>

struct TestType {};

int main()
{
    using std::any;
    using std::any_cast;

    any a;
    // expected-error@any:* {{static_assert failed "ValueType is required to be an rvalue reference or a CopyConstructible type"}}
    // expected-error@any:* {{non-const lvalue reference to type 'TestType' cannot bind to a temporary}}
    any_cast<TestType &>(std::move(a)); // expected-note {{requested here}}

    // expected-error@any:* {{static_assert failed "ValueType is required to be an rvalue reference or a CopyConstructible type"}}
    // expected-error@any:* {{non-const lvalue reference to type 'int' cannot bind to a temporary}}
    any_cast<int&>(42);
}
