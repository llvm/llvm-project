//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD cosh.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/math/coshf.h"
#include "src/mathvec/coshf.h"

using LlvmLibcCoshfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathvecTest<float, LIBC_NAMESPACE::coshf,
                                         LIBC_NAMESPACE::coshf>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcCoshfExhaustiveTest, EntireRange) { test_full_range_RN(); }
