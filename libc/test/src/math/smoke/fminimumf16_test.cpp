//===-- Unittests for fminimumf16 -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FMinimumTest.h"

#include "src/math/fminimumf16.h"

LIST_FMINIMUM_TESTS(float16, LIBC_NAMESPACE::fminimumf16)
