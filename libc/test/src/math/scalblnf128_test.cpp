//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Exhaustive tests for the scalblnf128 function.
///
//===----------------------------------------------------------------------===//

#include "ScalbnTest.h"

#include "src/__support/FPUtil/float128.h"
#include "src/math/scalblnf128.h"

#ifndef LIBC_TYPES_HAS_NATIVE_FLOAT128
using float128 = LIBC_NAMESPACE::fputil::Float128;
#endif // LIBC_TYPES_HAS_NATIVE_FLOAT128

namespace {
float128 wrapper(float128 x, int n) {
  return LIBC_NAMESPACE::scalblnf128(x, static_cast<long>(n));
}
} // namespace

LIST_SCALBN_TESTS(float128, wrapper)
