//===-- Unittests for fabsf16 ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FAbsTest.h"
#include "src/__support/FPUtil/float16.h"
#include "src/math/fabsf16.h"

#ifndef LIBC_TYPES_HAS_FLOAT16
using float16 = LIBC_NAMESPACE::fputil::Float16;
#endif // LIBC_TYPES_HAS_FLOAT16

LIST_FABS_TESTS(float16, LIBC_NAMESPACE::fabsf16)
