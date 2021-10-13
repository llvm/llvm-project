//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// XFAIL: !non-lockfree-atomics

// <atomic>

// template <class T>
//     bool
//     atomic_compare_exchange_weak_explicit(volatile atomic<T>*,
//                                           atomic<T>::value_type*,
//                                           atomic<T>::value_type,
//                                           memory_order, memory_order) noexcept;
//
// template <class T>
//     bool
//     atomic_compare_exchange_weak_explicit(atomic<T>*,
//                                           atomic<T>::value_type*,
//                                           atomic<T>::value_type,
//                                           memory_order, memory_order) noexcept;

#include <atomic>
#include <type_traits>
#include <cassert>

#include <cmpxchg_loop.h>

#include "test_macros.h"
#include "atomic_helpers.h"

template <class T>
struct TestFn {
  void operator()() const {
    {
        typedef std::atomic<T> A;
        T t(T(1));
        A a(t);
        assert(c_cmpxchg_weak_loop(&a, &t, T(2),
               std::memory_order_seq_cst, std::memory_order_seq_cst) == true);
        assert(a == T(2));
        assert(t == T(1));
        assert(std::atomic_compare_exchange_weak_explicit(&a, &t, T(3),
               std::memory_order_seq_cst, std::memory_order_seq_cst) == false);
        assert(a == T(2));
        assert(t == T(2));

        ASSERT_NOEXCEPT(std::atomic_compare_exchange_weak_explicit(&a, &t, T(3), std::memory_order_seq_cst,
                                                                   std::memory_order_seq_cst));
    }
    {
        typedef std::atomic<T> A;
        T t(T(1));
        volatile A a(t);
        assert(c_cmpxchg_weak_loop(&a, &t, T(2),
               std::memory_order_seq_cst, std::memory_order_seq_cst) == true);
        assert(a == T(2));
        assert(t == T(1));
        assert(std::atomic_compare_exchange_weak_explicit(&a, &t, T(3),
               std::memory_order_seq_cst, std::memory_order_seq_cst) == false);
        assert(a == T(2));
        assert(t == T(2));

        ASSERT_NOEXCEPT(std::atomic_compare_exchange_weak_explicit(&a, &t, T(3), std::memory_order_seq_cst,
                                                                   std::memory_order_seq_cst));
    }
  }
};

int main(int, char**)
{
    TestEachAtomicType<TestFn>()();

  return 0;
}
