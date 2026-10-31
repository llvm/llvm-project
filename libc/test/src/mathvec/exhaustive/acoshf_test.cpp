//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains exhaustive tests for single-precision SIMD acosh.
///
//===----------------------------------------------------------------------===//

#include "exhaustive_test.h"
#include "src/__support/CPP/simd.h"
#include "src/math/acoshf.h"
#include "src/mathvec/acoshf.h"

using LlvmLibcAcoshfExhaustiveTest =
    LlvmLibcUnaryOpExhaustiveMathvecTest<float, LIBC_NAMESPACE::acoshf,
                                         LIBC_NAMESPACE::acoshf>;

// Tests all possible 32-bit input patterns
TEST_F(LlvmLibcAcoshfExhaustiveTest, EntireRange) { test_full_range_RN(); }
