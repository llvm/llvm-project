//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for stdlib.h and complex.h not defining the <float.h> macros
///
//===----------------------------------------------------------------------===//

#include "include/complex.h"
#include "include/stdlib.h"
#ifdef LDBL_MANT_DIG
#error "<stdlib.h> or <complex.h> defines the <float.h> macros"
#endif
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStdlibNoFloatMacrosTest, FloatMacrosStayUndefined) {
  EXPECT_EQ(1, 1);
}
