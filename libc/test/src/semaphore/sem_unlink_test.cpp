//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sem_unlink.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/semaphore_macros.h"
#include "hdr/types/sem_t.h"
#include "src/semaphore/sem_close.h"
#include "src/semaphore/sem_open.h"
#include "src/semaphore/sem_post.h"
#include "src/semaphore/sem_unlink.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSemUnlinkTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSemUnlinkTest, UnlinkNonExistent) {
  const char *name = APPEND_LIBC_TEST("/llvmlibc_sem_missing_unlink");

  EXPECT_THAT(LIBC_NAMESPACE::sem_unlink(name), Fails(ENOENT));
}

TEST_F(LlvmLibcSemUnlinkTest, UnlinkRemovesName) {
  const char *name = APPEND_LIBC_TEST("/llvmlibc_sem_unlink_name");

  LIBC_NAMESPACE::sem_unlink(name);
  LIBC_NAMESPACE::libc_errno = 0;

  sem_t *sem = LIBC_NAMESPACE::sem_open(name, O_CREAT | O_EXCL, 0644, 1);
  ASSERT_NE(sem, SEM_FAILED);

  ASSERT_THAT(LIBC_NAMESPACE::sem_unlink(name), Succeeds());

  // The name is gone, so it can no longer be opened, and unlinking it again
  // reports ENOENT.
  EXPECT_EQ(LIBC_NAMESPACE::sem_open(name, 0), SEM_FAILED);
  ASSERT_ERRNO_EQ(ENOENT);
  LIBC_NAMESPACE::libc_errno = 0;

  EXPECT_THAT(LIBC_NAMESPACE::sem_unlink(name), Fails(ENOENT));

  // The semaphore itself outlives the name until the open handle is closed.
  ASSERT_THAT(LIBC_NAMESPACE::sem_post(sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_close(sem), Succeeds());
}
