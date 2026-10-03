//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD log10.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/math/log10f.h"
#include "src/mathvec/log10f.h"

using LlvmLibcLog10fExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathvecTest<float, LIBC_NAMESPACE::log10f,
                                         LIBC_NAMESPACE::log10f>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcLog10fExhaustiveTest, EntireRange) { test_full_range_RN(); }
