//===-- Unittests for llroundbf16 -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RoundToIntegerTest.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/llroundbf16.h"

LIST_ROUND_TO_INTEGER_TESTS(bfloat16, long long, LIBC_NAMESPACE::llroundbf16)
