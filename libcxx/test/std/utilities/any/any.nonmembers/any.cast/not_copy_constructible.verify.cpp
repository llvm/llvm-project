//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Throwing bad_any_cast is supported starting in macosx10.13
// UNSUPPORTED: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// <any>

// template <class ValueType>
// ValueType const any_cast(any const&);
//
// template <class ValueType>
// ValueType any_cast(any &);
//
// template <class ValueType>
// ValueType any_cast(any &&);

// Test instantiating the any_cast with a non-copyable type.

// This test only checks that we static_assert in any_cast when the
// constraints are not respected, however Clang will sometimes emit
// additional errors while trying to instantiate the rest of any_cast
// following the static_assert. We ignore unexpected errors in
// clang-verify to make the test more robust to changes in Clang.
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error

#include <any>

using std::any;
using std::any_cast;

struct no_copy
{
    no_copy() {}
    no_copy(no_copy &&) {}
    no_copy(no_copy const &) = delete;
};

struct no_move {
  no_move() {}
  no_move(no_move&&) = delete;
  no_move(no_move const&) {}
};

int main(int, char**) {
    any a;
    // expected-error-re@any:* {{static_assert failed{{.*}} "ValueType is required to be an lvalue reference or a CopyConstructible type"}}
    any_cast<no_copy>(static_cast<any&>(a)); // expected-note {{requested here}}

    // expected-error-re@any:* {{static_assert failed{{.*}} "ValueType is required to be a const lvalue reference or a CopyConstructible type"}}
    any_cast<no_copy>(static_cast<any const&>(a)); // expected-note {{requested here}}

    any_cast<no_copy>(static_cast<any &&>(a)); // OK

    // expected-error-re@any:* {{static_assert failed{{.*}} "ValueType is required to be an rvalue reference or a CopyConstructible type"}}
    any_cast<no_move>(static_cast<any &&>(a));

  return 0;
}
