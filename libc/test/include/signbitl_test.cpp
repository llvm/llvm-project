//===-- Unittests for signbit[l] macro ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SignbitTest.h"
#include "include/llvm-libc-macros/math-function-macros.h"

LIST_SIGNBIT_TESTS(long double, signbit)
