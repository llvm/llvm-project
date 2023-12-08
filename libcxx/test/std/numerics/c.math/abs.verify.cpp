//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cmath>

void f() {
    unsigned int ui = -5;
    (void)std::abs(ui); // expected-error {{call to 'abs' is ambiguous}}

    unsigned char uc = -5;
    (void)std::abs(uc); // expected-warning {{taking the absolute value of unsigned type 'unsigned char' has no effect}}

    unsigned short us = -5;
    (void)std::abs(us); // expected-warning {{taking the absolute value of unsigned type 'unsigned short' has no effect}}

    unsigned long ul = -5;
    (void)std::abs(ul); // expected-error {{call to 'abs' is ambiguous}}

    unsigned long long ull = -5;
    (void)std::abs(ull); // expected-error {{call to 'abs' is ambiguous}}
}
