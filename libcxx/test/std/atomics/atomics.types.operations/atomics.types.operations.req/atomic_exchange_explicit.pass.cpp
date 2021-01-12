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
//  ... assertion fails line 32

// <atomic>

// template <class T>
//     T
//     atomic_exchange_explicit(volatile atomic<T>* obj, T desr, memory_order m);
//
// template <class T>
//     T
//     atomic_exchange_explicit(atomic<T>* obj, T desr, memory_order m);

#include <atomic>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "atomic_helpers.h"

template <class T>
struct TestFn {
  void operator()() const {
    typedef std::atomic<T> A;
    A t;
    std::atomic_init(&t, T(1));
    assert(std::atomic_exchange_explicit(&t, T(2), std::memory_order_seq_cst)
           == T(1));
    assert(t == T(2));
    volatile A vt;
    std::atomic_init(&vt, T(3));
    assert(std::atomic_exchange_explicit(&vt, T(4), std::memory_order_seq_cst)
           == T(3));
    assert(vt == T(4));
  }
};


int main(int, char**)
{
    TestEachAtomicType<TestFn>()();

  return 0;
}
