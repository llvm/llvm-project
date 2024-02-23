//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: !has-128-bit-atomics

// <atomic>

// template <class T>
//     T
//     atomic_load_explicit(const volatile atomic<T>*, memory_order) noexcept;
//
// template <class T>
//     T
//     atomic_load_explicit(const atomic<T>*, memory_order) noexcept;

#include <atomic>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "atomic_helpers.h"

template <class T>
struct TestFn {
  void operator()() const {
    typedef std::atomic<T> A;
    A t(T(1));
    assert(std::atomic_load_explicit(&t, std::memory_order_seq_cst) == T(1));
    volatile A vt(T(2));
    assert(std::atomic_load_explicit(&vt, std::memory_order_seq_cst) == T(2));

    ASSERT_NOEXCEPT(std::atomic_load_explicit(&t, std::memory_order_seq_cst));
    ASSERT_NOEXCEPT(std::atomic_load_explicit(&vt, std::memory_order_seq_cst));
  }
};

int main(int, char**)
{
    TestEachAtomicType<TestFn>()();

  return 0;
}
