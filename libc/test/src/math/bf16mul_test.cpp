//===-- Unittests for bf16mul ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MulTest.h"

#include "src/math/bf16mul.h"

#include "src/__support/FPUtil/bfloat16.h"

LIST_MUL_TESTS(bfloat16, double, LIBC_NAMESPACE::bf16mul)
