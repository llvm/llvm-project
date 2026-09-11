//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD cbrt.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/math/cbrtf.h"
#include "src/mathvec/cbrtf.h"

using LlvmLibcCbrtfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathvecTest<float, LIBC_NAMESPACE::cbrtf,
                                         LIBC_NAMESPACE::cbrtf>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcCbrtfExhaustiveTest, EntireRange) { test_full_range_RN(); }
