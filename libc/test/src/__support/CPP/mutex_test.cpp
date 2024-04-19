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
struct Mutex {
    Mutex() : locked(false) {}

    void lock() {
        if (locked)
            // Sends signal 6.
            abort();
        locked = true;
    }

    void unlock() { 
        if (!locked)
            // Sends signal 6.
            abort();
        locked = false;
    }

    bool locked;
};

TEST(LlvmLibcMutexTest, Basic) {
    Mutex m;
    ASSERT_FALSE(m.locked);

    const int SIGABRT = 5;

    {
        lock_guard<Mutex> lg(m);
        ASSERT_TRUE(m.locked);
        ASSERT_DEATH([&](){ lock_guard<Mutex> lg2(m); }, SIGABRT);
    }

    ASSERT_FALSE(m.locked);
}
