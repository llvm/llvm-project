//===-- Unittests for canonicalizel ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CanonicalizeTest.h"

#include "src/math/canonicalizel.h"

LIST_CANONICALIZE_TESTS(long double, LIBC_NAMESPACE::canonicalizel)

#ifdef LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80

X86_80_SPECIAL_CANONICALIZE_TEST(long double, LIBC_NAMESPACE::canonicalizel)

#endif // LIBC_TYPES_LONG_DOUBLE_IS_X86_FLOAT80
