//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for umask.
///
//===----------------------------------------------------------------------===//

#include "hdr/sys_stat_macros.h"
#include "hdr/types/mode_t.h"
#include "src/sys/stat/umask.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/Test.h"

using LlvmLibcUmaskTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcUmaskTest, SmokeTest) {
  // umask() always succeeds and returns the previous mask. Set a mask, then
  // restore the original one and check that the mask we set is handed back.
  mode_t old_mask = LIBC_NAMESPACE::umask(S_IRWXG | S_IRWXO);
  EXPECT_EQ(LIBC_NAMESPACE::umask(old_mask),
            static_cast<mode_t>(S_IRWXG | S_IRWXO));
}
