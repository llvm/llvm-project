//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_any_cast is supported starting in macosx10.13
// UNSUPPORTED: use_system_cxx_lib && x86_64-apple-macosx10.12
// UNSUPPORTED: use_system_cxx_lib && x86_64-apple-macosx10.11
// UNSUPPORTED: use_system_cxx_lib && x86_64-apple-macosx10.10
// UNSUPPORTED: use_system_cxx_lib && x86_64-apple-macosx10.9

// <any>

// template <class ValueType>
// ValueType any_cast(any const &);

// Try and cast away const.

#include <any>

struct TestType {};
struct TestType2 {};

int main(int, char**)
{
    using std::any;
    using std::any_cast;

    any a;

    // expected-error@any:* {{drops 'const' qualifier}}
    // expected-error-re@any:* {{static_assert failed{{.*}} "ValueType is required to be a const lvalue reference or a CopyConstructible type"}}
    any_cast<TestType &>(static_cast<any const&>(a)); // expected-note {{requested here}}

    // expected-error@any:* {{cannot cast from lvalue of type 'const TestType' to rvalue reference type 'TestType &&'; types are not compatible}}
    // expected-error-re@any:* {{static_assert failed{{.*}} "ValueType is required to be a const lvalue reference or a CopyConstructible type"}}
    any_cast<TestType &&>(static_cast<any const&>(a)); // expected-note {{requested here}}

    // expected-error@any:* {{drops 'const' qualifier}}
    // expected-error-re@any:* {{static_assert failed{{.*}} "ValueType is required to be a const lvalue reference or a CopyConstructible type"}}
    any_cast<TestType2 &>(static_cast<any const&&>(a)); // expected-note {{requested here}}

    // expected-error@any:* {{cannot cast from lvalue of type 'const TestType2' to rvalue reference type 'TestType2 &&'; types are not compatible}}
    // expected-error-re@any:* {{static_assert failed{{.*}} "ValueType is required to be a const lvalue reference or a CopyConstructible type"}}
    any_cast<TestType2 &&>(static_cast<any const&&>(a)); // expected-note {{requested here}}

  return 0;
}
