//===-- Unittests for stdbit ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDSList-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "test/UnitTest/Test.h"

#include "llvm-libc-macros/stdbit-macros.h"
#include "src/__support/CPP/limits.h" // UINT_WIDTH
#include "src/stdbit/stdc_leading_zeros_uc.h"
#include "src/stdbit/stdc_leading_zeros_ui.h"
#include "src/stdbit/stdc_leading_zeros_ul.h"
#include "src/stdbit/stdc_leading_zeros_ull.h"
#include "src/stdbit/stdc_leading_zeros_us.h"

TEST(LlvmLibcStdbitTest, TypeGenericMacro) {
  using namespace LIBC_NAMESPACE;
  EXPECT_EQ(stdc_leading_zeros(0U), static_cast<unsigned>(UINT_WIDTH));
}
