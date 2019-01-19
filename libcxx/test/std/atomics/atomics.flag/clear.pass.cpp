//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
        std::atomic_flag f; // uninitialized
        f.clear();
        assert(f.test_and_set() == 0);
        f.clear();
        assert(f.test_and_set() == 0);
    }
    {
        std::atomic_flag f;
        f.clear(std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
        f.clear(std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
    }
    {
        std::atomic_flag f;
        f.clear(std::memory_order_release);
        assert(f.test_and_set() == 0);
        f.clear(std::memory_order_release);
        assert(f.test_and_set() == 0);
    }
    {
        std::atomic_flag f;
        f.clear(std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
        f.clear(std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
    }
    {
        volatile std::atomic_flag f;
        f.clear();
        assert(f.test_and_set() == 0);
        f.clear();
        assert(f.test_and_set() == 0);
    }
    {
        volatile std::atomic_flag f;
        f.clear(std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
        f.clear(std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
    }
    {
        volatile std::atomic_flag f;
        f.clear(std::memory_order_release);
        assert(f.test_and_set() == 0);
        f.clear(std::memory_order_release);
        assert(f.test_and_set() == 0);
    }
    {
        volatile std::atomic_flag f;
        f.clear(std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
        f.clear(std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
    }
}
