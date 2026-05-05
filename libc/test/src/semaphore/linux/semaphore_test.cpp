//===-- Unittests for the internal Semaphore class ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/fcntl_macros.h"
#include "src/semaphore/linux/semaphore.h"
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

// Named semaphore tests.

TEST(LlvmLibcSemaphoreTest, NamedOpenCloseUnlink) {
  const char *name = "/llvmlibc_test_sem";

  // clean up any leftover from previous test runs.
  Semaphore::unlink(name);

  // create a new named semaphore.
  auto result = Semaphore::open(name, O_CREAT | O_EXCL, 0644, 7);
  ASSERT_TRUE(result.has_value());

  Semaphore *sem = result.value();
  ASSERT_TRUE(sem->is_valid());
  ASSERT_EQ(sem->getvalue(), 7);

  // close and unlink.
  ASSERT_EQ(Semaphore::close(sem), 0);
  ASSERT_EQ(Semaphore::unlink(name), 0);
}

TEST(LlvmLibcSemaphoreTest, NamedOpenExisting) {
  const char *name = "/llvmlibc_test_sem_exist";

  Semaphore::unlink(name);

  // create a named semaphore.
  auto r1 = Semaphore::open(name, O_CREAT | O_EXCL, 0644, 10);
  ASSERT_TRUE(r1.has_value());

  Semaphore *sem1 = r1.value();
  ASSERT_EQ(sem1->getvalue(), 10);

  // open the same semaphore again without O_EXCL.
  auto r2 = Semaphore::open(name, O_CREAT, 0644, 99);
  ASSERT_TRUE(r2.has_value());

  Semaphore *sem2 = r2.value();
  ASSERT_EQ(sem2->getvalue(), 10);

  ASSERT_EQ(Semaphore::close(sem2), 0);
  ASSERT_EQ(Semaphore::close(sem1), 0);
  ASSERT_EQ(Semaphore::unlink(name), 0);
}

TEST(LlvmLibcSemaphoreTest, NamedOpenExclFails) {
  const char *name = "/llvmlibc_test_sem_excl";

  Semaphore::unlink(name);

  // create a named semaphore.
  auto r1 = Semaphore::open(name, O_CREAT | O_EXCL, 0644, 1);
  ASSERT_TRUE(r1.has_value());

  // trying O_CREAT | O_EXCL again should fail with EEXIST.
  auto r2 = Semaphore::open(name, O_CREAT | O_EXCL, 0644, 1);
  ASSERT_FALSE(r2.has_value());
  ASSERT_EQ(r2.error(), EEXIST);

  ASSERT_EQ(Semaphore::close(r1.value()), 0);
  ASSERT_EQ(Semaphore::unlink(name), 0);
}

TEST(LlvmLibcSemaphoreTest, NamedOpenNonExistent) {
  // opening a non-existent semaphore without O_CREAT should fail.
  auto result = Semaphore::open("/llvmlibc_nonexistent", 0, 0, 0);
  ASSERT_FALSE(result.has_value());
  ASSERT_EQ(result.error(), ENOENT);
}

TEST(LlvmLibcSemaphoreTest, NamedOpenInvalidName) {
  // empty name.
  auto r1 = Semaphore::open("", O_CREAT, 0644, 0);
  ASSERT_FALSE(r1.has_value());
  ASSERT_EQ(r1.error(), EINVAL);

  // name with embedded slash.
  auto r2 = Semaphore::open("/has/slash", O_CREAT, 0644, 0);
  ASSERT_FALSE(r2.has_value());
  ASSERT_EQ(r2.error(), EINVAL);

  // just a slash.
  auto r3 = Semaphore::open("/", O_CREAT, 0644, 0);
  ASSERT_FALSE(r3.has_value());
  ASSERT_EQ(r3.error(), EINVAL);
}
