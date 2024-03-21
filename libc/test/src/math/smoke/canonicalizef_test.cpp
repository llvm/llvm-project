//===-- Unittests for canonicalizef ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CanonicalizeTest.h"

#include "src/math/canonicalizef.h"

LIST_CANONICALIZE_TESTS(float, LIBC_NAMESPACE::canonicalizef)
