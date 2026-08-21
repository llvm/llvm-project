//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for named POSIX semaphores.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "hdr/semaphore_macros.h"
#include "hdr/types/sem_t.h"
#include "src/semaphore/sem_close.h"
#include "src/semaphore/sem_getvalue.h"
#include "src/semaphore/sem_open.h"
#include "src/semaphore/sem_post.h"
#include "src/semaphore/sem_unlink.h"
#include "src/semaphore/sem_wait.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSemOpenTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSemOpenTest, OpenCloseUnlink) {
  const char *name = APPEND_LIBC_TEST("/llvmlibc_sem_open");

  // Clear any leftover from a previous run
  LIBC_NAMESPACE::sem_unlink(name);
  LIBC_NAMESPACE::libc_errno = 0;

  sem_t *sem = LIBC_NAMESPACE::sem_open(name, O_CREAT | O_EXCL, 0644, 7);
  ASSERT_NE(sem, SEM_FAILED);
  ASSERT_ERRNO_SUCCESS();

  int value = -1;
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(sem, &value), Succeeds());
  EXPECT_EQ(value, 7);

  // A named semaphore supports the same operations as an unnamed one.
  ASSERT_THAT(LIBC_NAMESPACE::sem_wait(sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_post(sem), Succeeds());

  ASSERT_THAT(LIBC_NAMESPACE::sem_close(sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_unlink(name), Succeeds());
}

TEST_F(LlvmLibcSemOpenTest, OpenExisting) {
  const char *name = APPEND_LIBC_TEST("/llvmlibc_sem_open_existing");

  LIBC_NAMESPACE::sem_unlink(name);
  LIBC_NAMESPACE::libc_errno = 0;

  sem_t *first = LIBC_NAMESPACE::sem_open(name, O_CREAT | O_EXCL, 0644, 10);
  ASSERT_NE(first, SEM_FAILED);

  // The name already exists, so the value argument is ignored.
  sem_t *second = LIBC_NAMESPACE::sem_open(name, O_CREAT, 0644, 99);
  ASSERT_NE(second, SEM_FAILED);

  int value = -1;
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(second, &value), Succeeds());
  EXPECT_EQ(value, 10);

  ASSERT_THAT(LIBC_NAMESPACE::sem_close(second), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_close(first), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_unlink(name), Succeeds());
}

TEST_F(LlvmLibcSemOpenTest, DistinctHandlesShareState) {
  const char *name = APPEND_LIBC_TEST("/llvmlibc_sem_open_handles");

  LIBC_NAMESPACE::sem_unlink(name);
  LIBC_NAMESPACE::libc_errno = 0;

  sem_t *first = LIBC_NAMESPACE::sem_open(name, O_CREAT | O_EXCL, 0644, 0);
  ASSERT_NE(first, SEM_FAILED);

  sem_t *second = LIBC_NAMESPACE::sem_open(name, 0);
  ASSERT_NE(second, SEM_FAILED);

  // Each call returns its own handle.
  EXPECT_NE(first, second);

  // Both handles operate on the same underlying semaphore, so a post through
  // one is observed through the other.
  ASSERT_THAT(LIBC_NAMESPACE::sem_post(first), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_wait(second), Succeeds());

  // Closing one handle leaves the other usable.
  ASSERT_THAT(LIBC_NAMESPACE::sem_close(second), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_post(first), Succeeds());

  int value = -1;
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(first, &value), Succeeds());
  EXPECT_EQ(value, 1);

  ASSERT_THAT(LIBC_NAMESPACE::sem_close(first), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_unlink(name), Succeeds());
}

TEST_F(LlvmLibcSemOpenTest, OpenExclusiveTwiceFails) {
  const char *name = APPEND_LIBC_TEST("/llvmlibc_sem_open_excl");

  LIBC_NAMESPACE::sem_unlink(name);
  LIBC_NAMESPACE::libc_errno = 0;

  sem_t *sem = LIBC_NAMESPACE::sem_open(name, O_CREAT | O_EXCL, 0644, 1);
  ASSERT_NE(sem, SEM_FAILED);

  EXPECT_EQ(LIBC_NAMESPACE::sem_open(name, O_CREAT | O_EXCL, 0644, 1),
            SEM_FAILED);
  ASSERT_ERRNO_EQ(EEXIST);
  LIBC_NAMESPACE::libc_errno = 0;

  ASSERT_THAT(LIBC_NAMESPACE::sem_close(sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_unlink(name), Succeeds());
}

TEST_F(LlvmLibcSemOpenTest, OpenNonExistent) {
  const char *name = APPEND_LIBC_TEST("/llvmlibc_sem_missing");

  // Without O_CREAT the name must already exist.
  EXPECT_EQ(LIBC_NAMESPACE::sem_open(name, 0), SEM_FAILED);
  ASSERT_ERRNO_EQ(ENOENT);
}
