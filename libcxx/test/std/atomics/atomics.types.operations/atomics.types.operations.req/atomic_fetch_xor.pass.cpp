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

// template<class T>
//     T
//     atomic_fetch_xor(volatile atomic<T>*, atomic<T>::value_type) noexcept;
//
// template<class T>
//     T
//     atomic_fetch_xor(atomic<T>*, atomic<T>::value_type) noexcept;

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
        assert(std::atomic_fetch_xor(&t, T(2)) == T(1));
        assert(t == T(3));

        ASSERT_NOEXCEPT(std::atomic_fetch_xor(&t, T(2)));
    }
    {
        typedef std::atomic<T> A;
        volatile A t(T(3));
        assert(std::atomic_fetch_xor(&t, T(2)) == T(3));
        assert(t == T(1));

        ASSERT_NOEXCEPT(std::atomic_fetch_xor(&t, T(2)));
    }
  }
};

int main(int, char**)
{
    TestEachIntegralType<TestFn>()();

  return 0;
}
