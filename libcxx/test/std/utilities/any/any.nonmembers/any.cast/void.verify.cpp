//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <any>

// template<class T>
// const T* any_cast(const any* operand) noexcept;

// template<class T>
// T* any_cast(any* operand) noexcept;

#include <any>

void test() {
  {
    const std::any ca = 1;

    // expected-error-re@any:* {{static assertion failed{{.*}}_ValueType may not be void.}}
    std::any_cast<void>(&ca); // expected-note {{requested here}}
  }
  {
    std::any a = 1;

    // expected-error-re@any:* {{static assertion failed{{.*}}_ValueType may not be void.}}
    std::any_cast<void>(&a); // expected-note {{requested here}}
  }
}
