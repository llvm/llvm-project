//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD cos.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/__support/macros/optimization.h"
#ifdef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#define MATHVEC_TOL 1
#else
#define MATHVEC_TOL 0
#endif
// Keep the scalar reference correctly rounded in every build configuration.
#undef LIBC_MATH_HAS_SKIP_ACCURATE_PASS
#include "src/__support/math/cosf_double_eval.h"
#include "src/mathvec/cosf.h"

using LlvmLibcCosfExhaustiveTest = LlvmLibcUnaryOpExhaustiveMathvecTest<
    float, LIBC_NAMESPACE::math::double_eval::cosf, LIBC_NAMESPACE::cosf,
    MATHVEC_TOL>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcCosfExhaustiveTest, EntireRange) { test_full_range_RN(); }
