//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <any>

// template <class ValueType>
// ValueType const* any_cast(any const *) noexcept;

#include <any>

void test() {
  std::any a1       = 1;
  const std::any& a = a1;

  // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
  (void)std::any_cast<int&>(&a); // expected-note {{requested here}}

  // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
  (void)std::any_cast<int&&>(&a); // expected-note {{requested here}}

  // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
  (void)std::any_cast<int const&>(&a); // expected-note {{requested here}}

  // expected-error-re@any:* 1 {{static assertion failed{{.*}}_ValueType may not be a reference.}}
  (void)std::any_cast<int const&&>(&a); // expected-note {{requested here}}
}
