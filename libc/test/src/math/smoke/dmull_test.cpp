//===-- Unittests for dmull------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "FMulTest.h"

#include "src/math/dmull.h"

LIST_FMUL_TESTS(double, long double, LIBC_NAMESPACE::dmull)
