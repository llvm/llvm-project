//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// FIXME: When modules are enabled we can't affect the contents of <mutex>
// by defining a macro
// XFAIL: -fmodules

// <mutex>

// template <class ...Mutex> class lock_guard;

// explicit lock_guard(mutex_type& m);

#define _LIBCPP_ABI_VARIADIC_LOCK_GUARD
#include <mutex>
#include <cassert>

#include "test_macros.h"

struct TestMutex {
    bool locked = false;
    TestMutex() = default;
    ~TestMutex() { assert(!locked); }

    void lock() { assert(!locked); locked = true; }
    bool try_lock() { if (locked) return false; locked = true; return true; }
    void unlock() { assert(locked); locked = false; }

    TestMutex(TestMutex const&) = delete;
    TestMutex& operator=(TestMutex const&) = delete;
};

#if !defined(TEST_HAS_NO_EXCEPTIONS)
struct TestMutexThrows {
    bool locked = false;
    bool throws_on_lock = false;

    TestMutexThrows() = default;
    ~TestMutexThrows() { assert(!locked); }

    void lock() {
        assert(!locked);
        if (throws_on_lock) {
            throw 42;
        }
        locked = true;
    }

    bool try_lock() {
        if (locked) return false;
        lock();
        return true;
    }

    void unlock() { assert(locked); locked = false; }

    TestMutexThrows(TestMutexThrows const&) = delete;
    TestMutexThrows& operator=(TestMutexThrows const&) = delete;
};
#endif // !defined(TEST_HAS_NO_EXCEPTIONS)

int main()
{
    {
        using LG = std::lock_guard<>;
        LG lg;
    }
    {
        using LG = std::lock_guard<TestMutex, TestMutex>;
        TestMutex m1, m2;
        {
            LG lg(m1, m2);
            assert(m1.locked && m2.locked);
        }
        assert(!m1.locked && !m2.locked);
    }
    {
        using LG = std::lock_guard<TestMutex, TestMutex, TestMutex>;
        TestMutex m1, m2, m3;
        {
            LG lg(m1, m2, m3);
            assert(m1.locked && m2.locked && m3.locked);
        }
        assert(!m1.locked && !m2.locked && !m3.locked);
    }
#if !defined(TEST_HAS_NO_EXCEPTIONS)
    {
        using MT = TestMutexThrows;
        using LG = std::lock_guard<MT, MT>;
        MT m1, m2;
        m1.throws_on_lock = true;
        try {
            LG lg(m1, m2);
            assert(false);
        } catch (int) {}
        assert(!m1.locked && !m2.locked);
    }
    {
        using MT = TestMutexThrows;
        using LG = std::lock_guard<MT, MT, MT>;
        MT m1, m2, m3;
        m2.throws_on_lock = true;
        try {
            LG lg(m1, m2, m3);
            assert(false);
        } catch (int) {}
        assert(!m1.locked && !m2.locked && !m3.locked);
    }
#endif
}
