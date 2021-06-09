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

// template <class Integral>
//     Integral
//     atomic_fetch_add_explicit(volatile atomic<Integral>* obj, Integral op,
//                               memory_order m);
// template <class Integral>
//     Integral
//     atomic_fetch_add_explicit(atomic<Integral>* obj, Integral op,
//                               memory_order m);
// template <class T>
//     T*
//     atomic_fetch_add_explicit(volatile atomic<T*>* obj, ptrdiff_t op,
//                               memory_order m);
// template <class T>
//     T*
//     atomic_fetch_add_explicit(atomic<T*>* obj, ptrdiff_t op, memory_order m);

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
        A t(T(1));
        assert(std::atomic_fetch_add_explicit(&t, T(2),
                                            std::memory_order_seq_cst) == T(1));
        assert(t == T(3));
        ASSERT_NOEXCEPT(std::atomic_fetch_add_explicit(&t, 0, std::memory_order_relaxed));
    }
    {
        typedef std::atomic<T> A;
        volatile A t(T(1));
        assert(std::atomic_fetch_add_explicit(&t, T(2),
                                            std::memory_order_seq_cst) == T(1));
        assert(t == T(3));
        ASSERT_NOEXCEPT(std::atomic_fetch_add_explicit(&t, 0, std::memory_order_relaxed));
    }
  }
};

template <class T>
void
testp()
{
    {
        typedef std::atomic<T> A;
        typedef typename std::remove_pointer<T>::type X;
        X a[3] = {0};
        A t(&a[0]);
        assert(std::atomic_fetch_add_explicit(&t, 2, std::memory_order_seq_cst) == &a[0]);
        std::atomic_fetch_add_explicit<T>(&t, 0, std::memory_order_relaxed);
        assert(t == &a[2]);
        ASSERT_NOEXCEPT(std::atomic_fetch_add_explicit(&t, 0, std::memory_order_relaxed));
    }
    {
        typedef std::atomic<T> A;
        typedef typename std::remove_pointer<T>::type X;
        X a[3] = {0};
        volatile A t(&a[0]);
        assert(std::atomic_fetch_add_explicit(&t, 2, std::memory_order_seq_cst) == &a[0]);
        std::atomic_fetch_add_explicit<T>(&t, 0, std::memory_order_relaxed);
        assert(t == &a[2]);
        ASSERT_NOEXCEPT(std::atomic_fetch_add_explicit(&t, 0, std::memory_order_relaxed));
    }
}

int main(int, char**)
{
    TestEachIntegralType<TestFn>()();
    testp<int*>();
    testp<const int*>();

  return 0;
}
