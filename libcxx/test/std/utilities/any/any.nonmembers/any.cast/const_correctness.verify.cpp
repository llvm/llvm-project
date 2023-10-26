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
// ValueType any_cast(any const &);

// Try and cast away const.

// This test only checks that we static_assert in any_cast when the
// constraints are not respected, however Clang will sometimes emit
// additional errors while trying to instantiate the rest of any_cast
// following the static_assert. We ignore unexpected errors in
// clang-verify to make the test more robust to changes in Clang.
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error

#include <any>

struct TestType {};
struct TestType2 {};

void f() {
    std::any a;

    // expected-error-re@any:* {{static assertion failed{{.*}}ValueType is required to be a const lvalue reference or a CopyConstructible type}}
    std::any_cast<TestType &>(static_cast<std::any const&>(a)); // expected-note {{requested here}}

    // expected-error-re@any:* {{static assertion failed{{.*}}ValueType is required to be a const lvalue reference or a CopyConstructible type}}
    std::any_cast<TestType &&>(static_cast<std::any const&>(a)); // expected-note {{requested here}}

    // expected-error-re@any:* {{static assertion failed{{.*}}ValueType is required to be a const lvalue reference or a CopyConstructible type}}
    std::any_cast<TestType2 &>(static_cast<std::any const&&>(a)); // expected-note {{requested here}}

    // expected-error-re@any:* {{static assertion failed{{.*}}ValueType is required to be a const lvalue reference or a CopyConstructible type}}
    std::any_cast<TestType2 &&>(static_cast<std::any const&&>(a)); // expected-note {{requested here}}
}
