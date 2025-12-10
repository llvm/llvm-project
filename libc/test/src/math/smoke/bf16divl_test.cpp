//===-- Unittests for bf16divl --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DivTest.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/bf16divl.h"

LIST_DIV_TESTS(bfloat16, long double, LIBC_NAMESPACE::bf16divl)
