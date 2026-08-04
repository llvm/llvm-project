//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD acospi.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/math/acospif.h"
#include "src/mathvec/acospif.h"

using LlvmLibcAcospifExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathvecTest<float, LIBC_NAMESPACE::acospif,
                                         LIBC_NAMESPACE::acospif>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcAcospifExhaustiveTest, EntireRange) { test_full_range_RN(); }
