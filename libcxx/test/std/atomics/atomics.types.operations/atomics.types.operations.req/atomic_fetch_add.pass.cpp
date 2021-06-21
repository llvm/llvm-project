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
//     atomic_fetch_add(volatile atomic<Integral>* obj, Integral op);
//
// template <class Integral>
//     Integral
//     atomic_fetch_add(atomic<Integral>* obj, Integral op);
//
// template <class T>
//     T*
//     atomic_fetch_add(volatile atomic<T*>* obj, ptrdiff_t op);
//
// template <class T>
//     T*
//     atomic_fetch_add(atomic<T*>* obj, ptrdiff_t op);

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
        assert(std::atomic_fetch_add(&t, T(2)) == T(1));
        assert(t == T(3));
    }
    {
        typedef std::atomic<T> A;
        volatile A t(T(1));
        assert(std::atomic_fetch_add(&t, T(2)) == T(1));
        assert(t == T(3));
    }
  }
};

template <class T>
void testp()
{
    {
        typedef std::atomic<T> A;
        typedef typename std::remove_pointer<T>::type X;
        A t(T(1 * sizeof(X)));
        assert(std::atomic_fetch_add(&t, 2) == T(1*sizeof(X)));
#ifdef _LIBCPP_VERSION // libc++ is nonconforming
        std::atomic_fetch_add<X>(&t, 0);
#else
        std::atomic_fetch_add<T>(&t, 0);
#endif // _LIBCPP_VERSION
        assert(t == T(3*sizeof(X)));
    }
    {
        typedef std::atomic<T> A;
        typedef typename std::remove_pointer<T>::type X;
        volatile A t(T(1 * sizeof(X)));
        assert(std::atomic_fetch_add(&t, 2) == T(1*sizeof(X)));
#ifdef _LIBCPP_VERSION // libc++ is nonconforming
        std::atomic_fetch_add<X>(&t, 0);
#else
        std::atomic_fetch_add<T>(&t, 0);
#endif // _LIBCPP_VERSION
        assert(t == T(3*sizeof(X)));
    }
}

int main(int, char**)
{
    TestEachIntegralType<TestFn>()();
    testp<int*>();
    testp<const int*>();

  return 0;
}
