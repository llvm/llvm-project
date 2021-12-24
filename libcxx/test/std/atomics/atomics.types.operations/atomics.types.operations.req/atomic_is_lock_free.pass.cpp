//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: !is-lockfree-runtime-function

// <atomic>

// template <class T>
// bool atomic_is_lock_free(const volatile atomic<T>* obj) noexcept;
//
// template <class T>
// bool atomic_is_lock_free(const atomic<T>* obj) noexcept;

#include <atomic>
#include <cassert>

#include "test_macros.h"
#include "atomic_helpers.h"

template <class T>
struct TestFn {
  void operator()() const {
    typedef std::atomic<T> A;
    T t = T();
    A a(t);
    bool b1 = std::atomic_is_lock_free(static_cast<const A*>(&a));
    volatile A va(t);
    bool b2 = std::atomic_is_lock_free(static_cast<const volatile A*>(&va));
    assert(b1 == b2);

    ASSERT_NOEXCEPT(std::atomic_is_lock_free(static_cast<const A*>(&a)));
    ASSERT_NOEXCEPT(std::atomic_is_lock_free(static_cast<const volatile A*>(&va)));
  }
};

struct A {
  char x[4];
};

int main(int, char**) {
  TestFn<A>()();
  TestEachAtomicType<TestFn>()();
  return 0;
}
