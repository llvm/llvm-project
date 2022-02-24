//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// The following platforms have sizeof(long double) == sizeof(double), so this test doesn't apply to them.
// This test does apply to aarch64 where Arm's AAPCS64 is followed. There they are different sizes.
// UNSUPPORTED: target={{arm64|armv(7|8)(l|m)?|powerpc|powerpc64}}-{{.+}}
// UNSUPPORTED: target=x86_64-pc-windows-{{.+}}

// <compare>

// template<class T> constexpr strong_ordering strong_order(const T& a, const T& b);

// libc++ does not support strong_order(long double, long double) quite yet.
// This test verifies the error message we give for that case.
// TODO: remove this test once long double is properly supported.

#include <compare>

#include "test_macros.h"

void f() {
    long double ld = 3.14;
    (void)std::strong_order(ld, ld);  // expected-error@*:* {{std::strong_order is unimplemented for this floating-point type}}
}
