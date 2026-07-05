//===-- Unittests for bfloat16 multiplication -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MulTest.h"

#include "src/__support/FPUtil/bfloat16.h"

static bfloat16 mul_func(bfloat16 x, bfloat16 y) { return x * y; }

LIST_MUL_TESTS(bfloat16, bfloat16, mul_func)
