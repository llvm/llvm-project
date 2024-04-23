//===-- Unittests for canonicalizef128 ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CanonicalizeTest.h"

#include "src/math/canonicalizef128.h"

LIST_CANONICALIZE_TESTS(float128, LIBC_NAMESPACE::canonicalizef128)
