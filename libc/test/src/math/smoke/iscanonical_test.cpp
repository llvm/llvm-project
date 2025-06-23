//===-- Unittests for iscanonical -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IsCanonicalTest.h"

// The testing framework might include math.h and iscanonical macro definition
// in overlay mode.
#undef iscanonical

#include "src/math/iscanonical.h"

LIST_ISCANONICAL_TESTS(double, LIBC_NAMESPACE::iscanonical)
