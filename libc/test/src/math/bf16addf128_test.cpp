//===-- Unittests for bf16addf128 -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AddTest.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/bf16addf128.h"

LIST_ADD_TESTS(bfloat16, float128, LIBC_NAMESPACE::bf16addf128)
