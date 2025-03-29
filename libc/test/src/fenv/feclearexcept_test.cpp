//===-- Unittests for feclearexcept with exceptions enabled ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feclearexcept.h"

#include "src/__support/FPUtil/FEnvImpl.h"
#include "test/UnitTest/FEnvSafeTest.h"
#include "test/UnitTest/Test.h"

#include "hdr/fenv_macros.h"
#include <stdint.h>

#include "excepts.h"

using LlvmLibcFEnvTest = LIBC_NAMESPACE::testing::FEnvSafeTest;

TEST_F(LlvmLibcFEnvTest, ClearTest) {
  LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);

  for (int e : EXCEPTS)
    ASSERT_EQ(LIBC_NAMESPACE::fputil::test_except(e), 0);

  LIBC_NAMESPACE::fputil::raise_except(FE_ALL_EXCEPT);

  for (int e1 : EXCEPTS) {
    for (int e2 : EXCEPTS) {
      for (int e3 : EXCEPTS) {
        for (int e4 : EXCEPTS) {
          for (int e5 : EXCEPTS) {
            // We clear one exception and test to verify that it was cleared.
            LIBC_NAMESPACE::feclearexcept(e1 | e2 | e3 | e4 | e5);
            ASSERT_EQ(
                LIBC_NAMESPACE::fputil::test_except(e1 | e2 | e3 | e4 | e5), 0);
            // After clearing, we raise the exception again.
            LIBC_NAMESPACE::fputil::raise_except(e1 | e2 | e3 | e4 | e5);
          }
        }
      }
    }
  }
}
