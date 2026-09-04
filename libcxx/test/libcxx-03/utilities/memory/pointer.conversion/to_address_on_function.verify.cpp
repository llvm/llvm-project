//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T> constexpr T* __to_address(T* p) noexcept;
//     Mandates: T is not a function type.

#include <memory>

int f();

void test() {
    (void)std::__to_address(f); // expected-error@*:* {{is a function type}}
}
