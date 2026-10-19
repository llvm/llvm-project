//===-- Unittests for roundf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RoundTest.h"

#include "src/math/roundf16.h"

LIST_ROUND_TESTS(Roundf16, float16, LIBC_NAMESPACE::roundf16)
