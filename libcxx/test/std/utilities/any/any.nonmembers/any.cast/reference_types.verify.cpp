//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <any>

// template <class ValueType>
// ValueType const* any_cast(any const *) noexcept;
//
// template <class ValueType>
// ValueType * any_cast(any *) noexcept;

#include <any>

int main(int, char**)
{
    std::any a = 1;

    // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
    std::any_cast<int &>(&a); // expected-note {{requested here}}

    // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
    std::any_cast<int &&>(&a); // expected-note {{requested here}}

    // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
    std::any_cast<int const &>(&a); // expected-note {{requested here}}

    // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
    std::any_cast<int const&&>(&a); // expected-note {{requested here}}

    const std::any& a2 = a;

    // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
    std::any_cast<int &>(&a2); // expected-note {{requested here}}

    // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
    std::any_cast<int &&>(&a2); // expected-note {{requested here}}

    // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
    std::any_cast<int const &>(&a2); // expected-note {{requested here}}

    // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
    std::any_cast<int const &&>(&a2); // expected-note {{requested here}}

  return 0;
}
