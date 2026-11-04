//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Exhaustive test for the coshbf16 function.
///
//===----------------------------------------------------------------------===//

#include "src/__support/FPUtil/bfloat16.h"
#include "src/math/coshbf16.h"
#include "test/UnitTest/FPMatcher.h"
#include "test/UnitTest/Test.h"
#include "utils/MPFRWrapper/MPFRUtils.h"

using LlvmLibcCoshbf16Test = LIBC_NAMESPACE::testing::FPTest<bfloat16>;

namespace mpfr = LIBC_NAMESPACE::testing::mpfr;

// TODO: Add exhaustive PositiveRange / NegativeRange tests against MPFR once
// coshbf16 is implemented.
TEST_F(LlvmLibcCoshbf16Test, Placeholder) {}
