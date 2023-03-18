//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// UNSUPPORTED: availability-bad_any_cast-missing

// <any>

// template <class ValueType>
// ValueType any_cast(any &&);

// Try and use the rvalue any_cast to cast to an lvalue reference

#include <any>

struct TestType {};

void test_const_lvalue_cast_request_non_const_lvalue()
{
    const std::any a;
    // expected-error-re@any:* {{{{(static_assert|static assertion)}} failed{{.*}}ValueType is required to be a const lvalue reference or a CopyConstructible type}}
    // expected-error@any:* {{drops 'const' qualifier}}
    std::any_cast<TestType &>(a); // expected-note {{requested here}}

    const std::any a2(42);
    // expected-error-re@any:* {{{{(static_assert|static assertion)}} failed{{.*}}ValueType is required to be a const lvalue reference or a CopyConstructible type}}
    // expected-error@any:* {{drops 'const' qualifier}}
    std::any_cast<int&>(a2); // expected-note {{requested here}}
}

void test_lvalue_any_cast_request_rvalue()
{
    std::any a;
    // expected-error-re@any:* {{{{(static_assert|static assertion)}} failed{{.*}}ValueType is required to be an lvalue reference or a CopyConstructible type}}
    std::any_cast<TestType &&>(a); // expected-note {{requested here}}

    std::any a2(42);
    // expected-error-re@any:* {{{{(static_assert|static assertion)}} failed{{.*}}ValueType is required to be an lvalue reference or a CopyConstructible type}}
    std::any_cast<int&&>(a2); // expected-note {{requested here}}
}

void test_rvalue_any_cast_request_lvalue()
{
    std::any a;
    // expected-error-re@any:* {{{{(static_assert|static assertion)}} failed{{.*}}ValueType is required to be an rvalue reference or a CopyConstructible type}}
    // expected-error@any:* {{non-const lvalue reference to type 'TestType' cannot bind to a temporary}}
    std::any_cast<TestType &>(std::move(a)); // expected-note {{requested here}}

    // expected-error-re@any:* {{{{(static_assert|static assertion)}} failed{{.*}}ValueType is required to be an rvalue reference or a CopyConstructible type}}
    // expected-error@any:* {{non-const lvalue reference to type 'int' cannot bind to a temporary}}
    std::any_cast<int&>(42);
}

int main(int, char**)
{
    test_const_lvalue_cast_request_non_const_lvalue();
    test_lvalue_any_cast_request_rvalue();
    test_rvalue_any_cast_request_lvalue();

  return 0;
}
