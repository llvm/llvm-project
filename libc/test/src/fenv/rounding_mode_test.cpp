//===-- Unittests for fegetround and fesetround ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fegetround.h"
#include "src/fenv/fesetround.h"

#include "test/UnitTest/Test.h"

#include <fenv.h>

TEST(LlvmLibcRoundingModeTest, SetAndGet) {
  struct ResetDefaultRoundingMode {
    int original;
    ~ResetDefaultRoundingMode() {
      __llvm_libc::fesetround(original);
    }
  } reset{__llvm_libc::fegetround()};

  int s = __llvm_libc::fesetround(FE_TONEAREST);
  EXPECT_EQ(s, 0);
  int rm = __llvm_libc::fegetround();
  EXPECT_EQ(rm, FE_TONEAREST);

  s = __llvm_libc::fesetround(FE_UPWARD);
  EXPECT_EQ(s, 0);
  rm = __llvm_libc::fegetround();
  EXPECT_EQ(rm, FE_UPWARD);

  s = __llvm_libc::fesetround(FE_DOWNWARD);
  EXPECT_EQ(s, 0);
  rm = __llvm_libc::fegetround();
  EXPECT_EQ(rm, FE_DOWNWARD);

  s = __llvm_libc::fesetround(FE_TOWARDZERO);
  EXPECT_EQ(s, 0);
  rm = __llvm_libc::fegetround();
  EXPECT_EQ(rm, FE_TOWARDZERO);
}
