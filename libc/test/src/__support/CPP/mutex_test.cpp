//===-- Unittests for mutex -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/mutex.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::lock_guard;

// Simple class for testing cpp::lock_guard. It defines methods 'lock' and 
// 'unlock' which are required for the cpp::lock_guard class template.
class LockableObject {
    bool locked;

public:
    LockableObject() : locked(false) {}
    void lock() { locked = true; }
    void unlock() { locked = false; }
    bool is_locked() { return locked; }
};

TEST(LlvmLibcMutexTest, Basic) {
    LockableObject obj;
    ASSERT_FALSE(obj.is_locked());

    {
        lock_guard lg(obj);
        ASSERT_TRUE(obj.is_locked());
    }

    ASSERT_FALSE(obj.is_locked());
}
