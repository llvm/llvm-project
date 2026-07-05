//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: !has-64-bit-atomics

// <atomic>

// template<class T>
//     T
//     atomic_fetch_sub_explicit(volatile atomic<T>*, atomic<T>::difference_type,
//                               memory_order) noexcept;
//
// template<class T>
//     T
//     atomic_fetch_sub_explicit(atomic<T>*, atomic<T>::difference_type,
//                               memory_order) noexcept;

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
        A t(T(3));
        assert(std::atomic_fetch_sub_explicit(&t, T(2),
                                            std::memory_order_seq_cst) == T(3));
        assert(t == T(1));
        ASSERT_NOEXCEPT(std::atomic_fetch_sub_explicit(&t, 0, std::memory_order_relaxed));
    }
    {
        typedef std::atomic<T> A;
        volatile A t(T(3));
        assert(std::atomic_fetch_sub_explicit(&t, T(2),
                                            std::memory_order_seq_cst) == T(3));
        assert(t == T(1));
        ASSERT_NOEXCEPT(std::atomic_fetch_sub_explicit(&t, 0, std::memory_order_relaxed));
    }
  }
};

template <class T>
void testp()
{
    {
        typedef std::atomic<T> A;
        typedef typename std::remove_pointer<T>::type X;
        X a[3] = {0};
        A t(&a[2]);
        assert(std::atomic_fetch_sub_explicit(&t, 2, std::memory_order_seq_cst) == &a[2]);
        std::atomic_fetch_sub_explicit<T>(&t, 0, std::memory_order_relaxed);
        assert(t == &a[0]);
        ASSERT_NOEXCEPT(std::atomic_fetch_sub_explicit(&t, 0, std::memory_order_relaxed));
    }
    {
        typedef std::atomic<T> A;
        typedef typename std::remove_pointer<T>::type X;
        X a[3] = {0};
        volatile A t(&a[2]);
        assert(std::atomic_fetch_sub_explicit(&t, 2, std::memory_order_seq_cst) == &a[2]);
        std::atomic_fetch_sub_explicit<T>(&t, 0, std::memory_order_relaxed);
        assert(t == &a[0]);
        ASSERT_NOEXCEPT(std::atomic_fetch_sub_explicit(&t, 0, std::memory_order_relaxed));
    }
}

int main(int, char**)
{
    TestEachIntegralType<TestFn>()();
    testp<int*>();
    testp<const int*>();

  return 0;
}
