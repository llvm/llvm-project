//===-- Unittests for personality -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/personality/personality.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include <sys/personality.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSysPersonalityTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSysPersonalityTest, GetPersonality) {
  // Passing 0xffffffff retrieves the current persona without changing it.
  int persona = LIBC_NAMESPACE::personality(0xffffffff);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GE(persona, 0);
}

TEST_F(LlvmLibcSysPersonalityTest, SetAndRestore) {
  // Get the current persona.
  int original = LIBC_NAMESPACE::personality(0xffffffff);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_GE(original, 0);

  // Set the personality to the same value; should succeed and return the
  // previous persona.
  int prev = LIBC_NAMESPACE::personality(original);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(prev, original);

  // Verify the personality is unchanged.
  int current = LIBC_NAMESPACE::personality(0xffffffff);
  ASSERT_ERRNO_SUCCESS();
  ASSERT_EQ(current, original);
}
