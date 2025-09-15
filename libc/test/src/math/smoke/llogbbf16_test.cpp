//===-- Unittests for llogbbf16 -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ILogbTest.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/llogbbf16.h"

LIST_INTLOGB_TESTS(long, bfloat16, LIBC_NAMESPACE::llogbbf16);
