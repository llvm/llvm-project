//===-- Unittests for feupdateenv -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feupdateenv.h"

#include "src/__support/FPUtil/FEnvImpl.h"
#include "test/UnitTest/Test.h"

#include <fenv.h>
#include <signal.h>

TEST(LlvmLibcFEnvTest, UpdateEnvTest) {
  LIBC_NAMESPACE::fputil::disable_except(FE_ALL_EXCEPT);
  LIBC_NAMESPACE::fputil::clear_except(FE_ALL_EXCEPT);

  fenv_t env;
  ASSERT_EQ(LIBC_NAMESPACE::fputil::get_env(&env), 0);
  LIBC_NAMESPACE::fputil::set_except(FE_INVALID | FE_INEXACT);
  ASSERT_EQ(LIBC_NAMESPACE::feupdateenv(&env), 0);
  ASSERT_EQ(LIBC_NAMESPACE::fputil::test_except(FE_INVALID | FE_INEXACT),
            FE_INVALID | FE_INEXACT);
}
