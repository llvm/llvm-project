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

// bool atomic_flag_test_explicit(volatile atomic_flag*, memory_order);
// bool atomic_flag_test_explicit(atomic_flag*, memory_order);

#include <atomic>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_relaxed) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_relaxed) == 1);
    }
    {
        std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_consume) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_consume) == 1);
    }
    {
        std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_acquire) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_acquire) == 1);
    }
#ifdef _LIBCPP_VERSION // Don't violate precondition [atomics.flag]/6
    {
        std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_release) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_release) == 1);
    }
    {
        std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_acq_rel) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_acq_rel) == 1);
    }
#endif // _LIBCPP_VERSION
    {
        std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_seq_cst) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_seq_cst) == 1);
    }
    {
        volatile std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_relaxed) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_relaxed) == 1);
    }
    {
        volatile std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_consume) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_consume) == 1);
    }
    {
        volatile std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_acquire) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_acquire) == 1);
    }
#ifdef _LIBCPP_VERSION // Don't violate precondition [atomics.flag]/6
    {
        volatile std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_release) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_release) == 1);
    }
    {
        volatile std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_acq_rel) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_acq_rel) == 1);
    }
#endif // _LIBCPP_VERSION
    {
        volatile std::atomic_flag f;
        f.clear();
        assert(atomic_flag_test_explicit(&f, std::memory_order_seq_cst) == 0);
        assert(f.test_and_set() == 0);
        assert(atomic_flag_test_explicit(&f, std::memory_order_seq_cst) == 1);
    }

    return 0;
}
