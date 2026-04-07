//===-- Unittests for POSIX semaphores -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "src/semaphore/sem_destroy.h"
#include "src/semaphore/sem_getvalue.h"
#include "src/semaphore/sem_init.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <limits.h>
#include <semaphore.h>

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;

namespace {

using LlvmLibcSemaphoreTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

} // namespace

TEST_F(LlvmLibcSemaphoreTest, PublicSurface) {
  sem_t sem = {};
  (void)sem;
  ASSERT_GE(SEM_VALUE_MAX, 32767);
}

TEST_F(LlvmLibcSemaphoreTest, UnnamedLifecycle) {
  sem_t sem;
  int value = -1;

  ASSERT_THAT(LIBC_NAMESPACE::sem_init(&sem, 0, 3), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Succeeds());
  ASSERT_EQ(value, 3);

  ASSERT_THAT(LIBC_NAMESPACE::sem_destroy(&sem), Succeeds());
  ASSERT_THAT(LIBC_NAMESPACE::sem_getvalue(&sem, &value), Fails(EINVAL));
}

TEST_F(LlvmLibcSemaphoreTest, UnnamedInitRejectsTooLargeValue) {
  sem_t sem;
  ASSERT_THAT(LIBC_NAMESPACE::sem_init(
                  &sem, 0, static_cast<unsigned int>(SEM_VALUE_MAX) + 1U),
              Fails(EINVAL));
}
