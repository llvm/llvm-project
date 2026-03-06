//===-- Unittests for bfloat16 subtraction --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubTest.h"

#include "src/__support/FPUtil/bfloat16.h"

static bfloat16 sub_func(bfloat16 x, bfloat16 y) { return x - y; }

LIST_SUB_TESTS(bfloat16, bfloat16, sub_func)
