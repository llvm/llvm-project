//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getresuid.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/unistd/geteuid.h"
#include "src/unistd/getresuid.h"
#include "src/unistd/getuid.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcGetResUidTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetResUidTest, GetCurrentResUid) {
  uid_t ruid = static_cast<uid_t>(-1);
  uid_t euid = static_cast<uid_t>(-1);
  uid_t suid = static_cast<uid_t>(-1);
  ASSERT_THAT(LIBC_NAMESPACE::getresuid(&ruid, &euid, &suid), Succeeds());
  EXPECT_EQ(ruid, LIBC_NAMESPACE::getuid());
  EXPECT_EQ(euid, LIBC_NAMESPACE::geteuid());
  EXPECT_NE(suid, static_cast<uid_t>(-1));
}

TEST_F(LlvmLibcGetResUidTest, InvalidPointer) {
  ASSERT_THAT(LIBC_NAMESPACE::getresuid(nullptr, nullptr, nullptr),
              Fails(EFAULT));
}
