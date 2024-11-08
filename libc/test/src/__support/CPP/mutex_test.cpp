//===-- Unittests for mutex -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/mutex.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::adopt_lock;
using LIBC_NAMESPACE::cpp::lock_guard;

// Simple struct for testing cpp::lock_guard. It defines methods 'lock' and
// 'unlock' which are required for the cpp::lock_guard class template.
struct Mutex {
  // Flag to show whether this mutex is locked.
  bool locked = false;

  // Flag to show if this mutex has been double locked.
  bool double_locked = false;

  // Flag to show if this mutex has been double unlocked.
  bool double_unlocked = false;

  Mutex() {}

  void lock() {
    if (locked)
      double_locked = true;

    locked = true;
  }

  void unlock() {
    if (!locked)
      double_unlocked = true;

    locked = false;
  }
};

TEST(LlvmLibcMutexTest, Basic) {
  Mutex m;
  ASSERT_FALSE(m.locked);
  ASSERT_FALSE(m.double_locked);
  ASSERT_FALSE(m.double_unlocked);

  {
    lock_guard lg(m);
    ASSERT_TRUE(m.locked);
    ASSERT_FALSE(m.double_locked);
  }

  ASSERT_FALSE(m.locked);
  ASSERT_FALSE(m.double_unlocked);
}

TEST(LlvmLibcMutexTest, AcquireLocked) {
  Mutex m;
  ASSERT_FALSE(m.locked);
  ASSERT_FALSE(m.double_locked);
  ASSERT_FALSE(m.double_unlocked);

  // Lock the mutex before placing a lock guard on it.
  m.lock();
  ASSERT_TRUE(m.locked);
  ASSERT_FALSE(m.double_locked);

  {
    lock_guard lg(m, adopt_lock);
    ASSERT_TRUE(m.locked);
    ASSERT_FALSE(m.double_locked);
  }

  ASSERT_FALSE(m.locked);
  ASSERT_FALSE(m.double_unlocked);
}
