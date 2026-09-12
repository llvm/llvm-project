//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for sem_getvalue.
///
//===----------------------------------------------------------------------===//

#include "hdr/limits_macros.h"
#include "hdr/types/sem_t.h"
#include "src/semaphore/sem_destroy.h"
#include "src/semaphore/sem_getvalue.h"
#include "src/semaphore/sem_init.h"
#include "src/semaphore/sem_post.h"
#include "src/semaphore/sem_wait.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcSemGetValueTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcSemGetValueTest, ReportsInitialValue) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 3), Succeeds());

  int value = -1;
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Succeeds());
  EXPECT_EQ(value, 3);

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemGetValueTest, ReportsMaximumValue) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(
                  &sem, 0, static_cast<unsigned int>(SEM_VALUE_MAX)),
              Succeeds());

  int value = -1;
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Succeeds());
  EXPECT_EQ(value, SEM_VALUE_MAX);

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}

TEST_F(LlvmLibcSemGetValueTest, TracksPostAndWait) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 0), Succeeds());

  int value = -1;
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Succeeds());
  EXPECT_EQ(value, 0);

  ASSERT_THAT(LIBC_NAMESPACE::sem_post(&sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_post(&sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Succeeds());
  EXPECT_EQ(value, 2);

  ASSERT_THAT(LIBC_NAMESPACE::sem_wait(&sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Succeeds());
  EXPECT_EQ(value, 1);

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
}
