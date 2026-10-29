//===-- Unittests for fabsbf16 --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains tests for fabsbf16
///
//===----------------------------------------------------------------------===//

#include "FAbsTest.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/macros/properties/types.h"
#include "src/math/fabsbf16.h"

#if defined(LIBC_TYPES_HAS_BUILTIN_BFLOAT16)
static bfloat16 fabsbf16_test_wrapper(bfloat16 x) {
  return bfloat16(LIBC_NAMESPACE::fabsbf16(static_cast<__bf16>(x)));
}
LIST_FABS_TESTS(Fabsbf16, bfloat16, fabsbf16_test_wrapper)
#else
LIST_FABS_TESTS(Fabsbf16, bfloat16, LIBC_NAMESPACE::fabsbf16)
#endif
