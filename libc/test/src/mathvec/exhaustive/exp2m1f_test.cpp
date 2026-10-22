//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD exp2m1.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/math/exp2m1f.h"
#include "src/mathvec/exp2m1f.h"

using LlvmLibcExp2m1fExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathvecTest<float, LIBC_NAMESPACE::exp2m1f,
                                         LIBC_NAMESPACE::exp2m1f>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcExp2m1fExhaustiveTest, EntireRange) { test_full_range_RN(); }
