//===-- Unittests for llogbl ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "src/math/llogbl.h"

LIST_INTLOGB_TESTS(long, long double, LIBC_NAMESPACE::llogbl);

// Constexpr tests: verify that llogbl can be evaluated at compile time.
// These static_assert cases cover normal numbers with various exponents.
namespace {

// Normal numbers: 2^0 = 1.0 => exponent 0
static_assert(LIBC_NAMESPACE::llogbl(1.0L) == 0);
static_assert(LIBC_NAMESPACE::llogbl(-1.0L) == 0);

// Normal numbers: 2^1 = 2.0 => exponent 1
static_assert(LIBC_NAMESPACE::llogbl(2.0L) == 1);
static_assert(LIBC_NAMESPACE::llogbl(-2.0L) == 1);

// Normal numbers: 2^2 = 4.0 => exponent 2
static_assert(LIBC_NAMESPACE::llogbl(4.0L) == 2);
static_assert(LIBC_NAMESPACE::llogbl(-4.0L) == 2);

// Normal numbers: 2^(-1) = 0.5 => exponent -1
static_assert(LIBC_NAMESPACE::llogbl(0.5L) == -1);
static_assert(LIBC_NAMESPACE::llogbl(-0.5L) == -1);

// Normal numbers: 2^3 = 8.0 => exponent 3
static_assert(LIBC_NAMESPACE::llogbl(8.0L) == 3);

// Normal numbers: 2^10 = 1024.0 => exponent 10
static_assert(LIBC_NAMESPACE::llogbl(1024.0L) == 10);
} // anonymous namespace
