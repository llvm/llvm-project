//===-- Unittests for bf16add ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AddTest.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/bf16add.h"

LIST_ADD_TESTS(bfloat16, double, LIBC_NAMESPACE::bf16add)
