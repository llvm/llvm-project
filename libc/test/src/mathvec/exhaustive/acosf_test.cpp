//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD acos.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/math/acosf.h"
#include "src/mathvec/acosf.h"

using LlvmLibcAcosfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathvecTest<float, LIBC_NAMESPACE::acosf,
                                         LIBC_NAMESPACE::acosf>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcAcosfExhaustiveTest, EntireRange) { test_full_range_RN(); }
