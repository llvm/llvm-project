//===-- Unittests for lroundf16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RoundToIntegerTest.h"

#include "src/math/lroundf16.h"

LIST_ROUND_TO_INTEGER_TESTS(float16, long, LIBC_NAMESPACE::lroundf16)
