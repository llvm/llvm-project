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

// lock_guard(Mutex&..., adopt_lock_t);

#define _LIBCPP_ABI_VARIADIC_LOCK_GUARD
#include <mutex>
#include <cassert>

struct TestMutex {
    bool locked = false;
    TestMutex() = default;

    void lock() { assert(!locked); locked = true; }
    bool try_lock() { if (locked) return false; locked = true; return true; }
    void unlock() { assert(locked); locked = false; }

    TestMutex(TestMutex const&) = delete;
    TestMutex& operator=(TestMutex const&) = delete;
};

int main()
{
    {
        using LG = std::lock_guard<>;
        LG lg(std::adopt_lock);
    }
    {
        TestMutex m1, m2;
        using LG = std::lock_guard<TestMutex, TestMutex>;
        m1.lock(); m2.lock();
        {
            LG lg(m1, m2, std::adopt_lock);
            assert(m1.locked && m2.locked);
        }
        assert(!m1.locked && !m2.locked);
    }
    {
        TestMutex m1, m2, m3;
        using LG = std::lock_guard<TestMutex, TestMutex, TestMutex>;
        m1.lock(); m2.lock(); m3.lock();
        {
            LG lg(m1, m2, m3, std::adopt_lock);
            assert(m1.locked && m2.locked && m3.locked);
        }
        assert(!m1.locked && !m2.locked && !m3.locked);
    }

}
