//===-- Unittests for the internal Semaphore class ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/semaphore/posix_semaphore.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::Semaphore;

TEST(LlvmLibcSemaphoreTest, InitAndGetValue) {
  Semaphore sem(3);
  ASSERT_TRUE(sem.is_valid());
  ASSERT_EQ(sem.getvalue(), 3);
}

TEST(LlvmLibcSemaphoreTest, Destroy) {
  Semaphore sem(5);
  ASSERT_TRUE(sem.is_valid());
  sem.destroy();
  ASSERT_FALSE(sem.is_valid());
}
