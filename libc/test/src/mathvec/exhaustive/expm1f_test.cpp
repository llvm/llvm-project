//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD expm1.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/math/expm1f.h"
#include "src/mathvec/expm1f.h"

using LlvmLibcExpm1fExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathvecTest<float, LIBC_NAMESPACE::expm1f,
                                         LIBC_NAMESPACE::expm1f>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcExpm1fExhaustiveTest, EntireRange) { test_full_range_RN(); }
