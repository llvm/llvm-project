//===-- Unittests for bf16div ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DivTest.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/bf16div.h"

LIST_DIV_TESTS(bfloat16, double, LIBC_NAMESPACE::bf16div)
