//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class T> constexpr T* to_address(T* p) noexcept;
//     Mandates: T is not a function type.

#include <memory>

int (*pf)();

void test() {
    (void)std::to_address(pf); // expected-error@*:* {{is a function type}}
}
