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

// template <class T>
//     bool
//     atomic_is_lock_free(const volatile atomic<T>* obj);
//
// template <class T>
//     bool
//     atomic_is_lock_free(const atomic<T>* obj);

#include <atomic>
#include <cassert>

#include "atomic_helpers.h"

template <class T>
struct TestFn {
  void operator()() const {
    typedef std::atomic<T> A;
    A t;
    bool b1 = std::atomic_is_lock_free(static_cast<const A*>(&t));
    volatile A vt;
    bool b2 = std::atomic_is_lock_free(static_cast<const volatile A*>(&vt));
    assert(b1 == b2);
  }
};

struct A
{
    char _[4];
};

int main()
{
    TestFn<A>()();
    TestEachAtomicType<TestFn>()();
}
