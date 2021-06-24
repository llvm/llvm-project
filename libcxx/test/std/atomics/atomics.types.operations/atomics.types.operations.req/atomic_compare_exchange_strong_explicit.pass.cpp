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
//  ... assertion fails line 38

// <atomic>

// template <class T>
//     bool
//     atomic_compare_exchange_strong_explicit(volatile atomic<T>* obj, T* expc,
//                                           T desr,
//                                           memory_order s, memory_order f);
//
// template <class T>
//     bool
//     atomic_compare_exchange_strong_explicit(atomic<T>* obj, T* expc, T desr,
//                                           memory_order s, memory_order f);

#include <atomic>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "atomic_helpers.h"

template <class T>
struct TestFn {
  void operator()() const {
    {
        typedef std::atomic<T> A;
        T t(T(1));
        A a(t);
        assert(std::atomic_compare_exchange_strong_explicit(&a, &t, T(2),
               std::memory_order_seq_cst, std::memory_order_seq_cst) == true);
        assert(a == T(2));
        assert(t == T(1));
        assert(std::atomic_compare_exchange_strong_explicit(&a, &t, T(3),
               std::memory_order_seq_cst, std::memory_order_seq_cst) == false);
        assert(a == T(2));
        assert(t == T(2));
    }
    {
        typedef std::atomic<T> A;
        T t(T(1));
        volatile A a(t);
        assert(std::atomic_compare_exchange_strong_explicit(&a, &t, T(2),
               std::memory_order_seq_cst, std::memory_order_seq_cst) == true);
        assert(a == T(2));
        assert(t == T(1));
        assert(std::atomic_compare_exchange_strong_explicit(&a, &t, T(3),
               std::memory_order_seq_cst, std::memory_order_seq_cst) == false);
        assert(a == T(2));
        assert(t == T(2));
    }
  }
};

int main(int, char**)
{
    TestEachAtomicType<TestFn>()();

  return 0;
}
