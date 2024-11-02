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
//     void
//     atomic_store(volatile atomic<T>*, atomic<T>::value_type) noexcept;
//
// template <class T>
//     void
//     atomic_store(atomic<T>*, atomic<T>::value_type) noexcept;

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
    std::atomic_store(&t, T(1));
    assert(t == T(1));
    volatile A vt;
    std::atomic_store(&vt, T(2));
    assert(vt == T(2));

    ASSERT_NOEXCEPT(std::atomic_store(&t, T(1)));
    ASSERT_NOEXCEPT(std::atomic_store(&vt, T(2)));
  }
};


int main(int, char**)
{
    TestEachAtomicType<TestFn>()();

  return 0;
}
