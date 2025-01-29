//===-- Unittests for dsqrtl ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SqrtTest.h"

#include "src/math/dsqrtl.h"

LIST_NARROWING_SQRT_TESTS(double, long double, LIBC_NAMESPACE::dsqrtl)
