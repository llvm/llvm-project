//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD exp10.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/math/exp10f_double_eval.h"
#include "src/mathvec/exp10f.h"

using LlvmLibcExp10fExhaustiveTest = LlvmLibcUnaryOpExhaustiveMathvecTest<
    float, LIBC_NAMESPACE::math::double_eval::exp10f, LIBC_NAMESPACE::exp10f>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcExp10fExhaustiveTest, EntireRange) { test_full_range_RN(); }
