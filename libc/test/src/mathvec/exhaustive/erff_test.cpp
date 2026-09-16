//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD erf.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/math/erff.h"
#include "src/mathvec/erff.h"

using LlvmLibcErffExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathvecTest<float, LIBC_NAMESPACE::erff,
                                         LIBC_NAMESPACE::erff>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcErffExhaustiveTest, EntireRange) { test_full_range_RN(); }
