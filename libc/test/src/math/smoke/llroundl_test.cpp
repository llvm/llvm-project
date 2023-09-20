//===-- Unittests for llroundl --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RoundToIntegerTest.h"

#include "src/math/llroundl.h"

LIST_ROUND_TO_INTEGER_TESTS(long double, long long, __llvm_libc::llroundl)
