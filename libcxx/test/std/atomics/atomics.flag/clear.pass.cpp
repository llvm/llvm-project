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

// <atomic>

// struct atomic_flag

// void clear(memory_order = memory_order_seq_cst);
// void clear(memory_order = memory_order_seq_cst) volatile;

#include <atomic>
#include <cassert>

int main()
{
    {
        std::atomic_flag f;
        f.test_and_set();
        f.clear();
        assert(f.test_and_set() == 0);
    }
    {
        std::atomic_flag f;
        f.test_and_set();
        f.clear(std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
    }
    {
        std::atomic_flag f;
        f.test_and_set();
        f.clear(std::memory_order_release);
        assert(f.test_and_set() == 0);
    }
    {
        std::atomic_flag f;
        f.test_and_set();
        f.clear(std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
    }
    {
        volatile std::atomic_flag f;
        f.test_and_set();
        f.clear();
        assert(f.test_and_set() == 0);
    }
    {
        volatile std::atomic_flag f;
        f.test_and_set();
        f.clear(std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
    }
    {
        volatile std::atomic_flag f;
        f.test_and_set();
        f.clear(std::memory_order_release);
        assert(f.test_and_set() == 0);
    }
    {
        volatile std::atomic_flag f;
        f.test_and_set();
        f.clear(std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
    }
}
