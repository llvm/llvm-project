//===-- Unittests for bf16mulf --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MulTest.h"

#include "src/math/bf16mulf.h"

#include "src/__support/FPUtil/bfloat16.h"

LIST_MUL_TESTS(bfloat16, float, LIBC_NAMESPACE::bf16mulf)
