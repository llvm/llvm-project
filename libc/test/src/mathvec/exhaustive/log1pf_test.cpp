//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD log1p.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/math/log1pf.h"
#include "src/mathvec/log1pf.h"

using LlvmLibcLog1pfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathvecTest<float, LIBC_NAMESPACE::log1pf,
                                         LIBC_NAMESPACE::log1pf>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcLog1pfExhaustiveTest, EntireRange) { test_full_range_RN(); }
