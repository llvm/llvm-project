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

// void atomic_flag_clear_explicit(volatile atomic_flag*, memory_order);
// void atomic_flag_clear_explicit(atomic_flag*, memory_order);

#include <atomic>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::atomic_flag f; // uninitialized first
        atomic_flag_clear_explicit(&f, std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
        atomic_flag_clear_explicit(&f, std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
    }
    {
        std::atomic_flag f;
        atomic_flag_clear_explicit(&f, std::memory_order_release);
        assert(f.test_and_set() == 0);
        atomic_flag_clear_explicit(&f, std::memory_order_release);
        assert(f.test_and_set() == 0);
    }
    {
        std::atomic_flag f;
        atomic_flag_clear_explicit(&f, std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
        atomic_flag_clear_explicit(&f, std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
    }
    {
        volatile std::atomic_flag f;
        atomic_flag_clear_explicit(&f, std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
        atomic_flag_clear_explicit(&f, std::memory_order_relaxed);
        assert(f.test_and_set() == 0);
    }
    {
        volatile std::atomic_flag f;
        atomic_flag_clear_explicit(&f, std::memory_order_release);
        assert(f.test_and_set() == 0);
        atomic_flag_clear_explicit(&f, std::memory_order_release);
        assert(f.test_and_set() == 0);
    }
    {
        volatile std::atomic_flag f;
        atomic_flag_clear_explicit(&f, std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
        atomic_flag_clear_explicit(&f, std::memory_order_seq_cst);
        assert(f.test_and_set() == 0);
    }

  return 0;
}
