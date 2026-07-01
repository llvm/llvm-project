//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// XFAIL: !has-64-bit-atomics

// <atomic>

// Test all atomic_store_* non-member functions (non-explicit versions):
//   atomic_store_add, atomic_store_sub, atomic_store_and, atomic_store_or, atomic_store_xor
//
// template<class T>
//     void atomic_store_add(volatile atomic<T>*, atomic<T>::difference_type) noexcept;
// template<class T>
//     void atomic_store_add(atomic<T>*, atomic<T>::difference_type) noexcept;
//
// (Same pattern for store_sub, store_and, store_or, store_xor)

#include <atomic>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "atomic_store_operations_helper.h"

template <class T>
struct TestFn {
  void operator()() const {
    {
      typedef std::atomic<T> A;
      A t;

      auto load      = [&]() { return t.load(); };
      auto store     = [&](T val) { t.store(val); };
      auto store_add = [&](T val, auto) { std::atomic_store_add(&t, val); };
      auto store_sub = [&](T val, auto) { std::atomic_store_sub(&t, val); };
      auto store_and = [&](T val, auto) { std::atomic_store_and(&t, val); };
      auto store_or  = [&](T val, auto) { std::atomic_store_or(&t, val); };
      auto store_xor = [&](T val, auto) { std::atomic_store_xor(&t, val); };

      test_store_operations_integral<T>(load, store, store_add, store_sub, store_and, store_or, store_xor);
      ASSERT_NOEXCEPT(std::atomic_store_add(&t, T(0)));
      ASSERT_NOEXCEPT(std::atomic_store_sub(&t, T(0)));
      ASSERT_NOEXCEPT(std::atomic_store_and(&t, T(0)));
      ASSERT_NOEXCEPT(std::atomic_store_or(&t, T(0)));
      ASSERT_NOEXCEPT(std::atomic_store_xor(&t, T(0)));
    }
    {
      typedef std::atomic<T> A;
      volatile A t;

      auto load      = [&]() { return t.load(); };
      auto store     = [&](T val) { t.store(val); };
      auto store_add = [&](T val, auto) { std::atomic_store_add(&t, val); };
      auto store_sub = [&](T val, auto) { std::atomic_store_sub(&t, val); };
      auto store_and = [&](T val, auto) { std::atomic_store_and(&t, val); };
      auto store_or  = [&](T val, auto) { std::atomic_store_or(&t, val); };
      auto store_xor = [&](T val, auto) { std::atomic_store_xor(&t, val); };

      test_store_operations_integral<T>(load, store, store_add, store_sub, store_and, store_or, store_xor);
      ASSERT_NOEXCEPT(std::atomic_store_add(&t, T(0)));
      ASSERT_NOEXCEPT(std::atomic_store_sub(&t, T(0)));
      ASSERT_NOEXCEPT(std::atomic_store_and(&t, T(0)));
      ASSERT_NOEXCEPT(std::atomic_store_or(&t, T(0)));
      ASSERT_NOEXCEPT(std::atomic_store_xor(&t, T(0)));
    }
  }
};

template <class T>
void testp() {
  {
    typedef std::atomic<T> A;
    A t;

    auto load      = [&]() { return t.load(); };
    auto store     = [&](T val) { t.store(val); };
    auto store_add = [&](std::ptrdiff_t val, auto) { std::atomic_store_add(&t, val); };
    auto store_sub = [&](std::ptrdiff_t val, auto) { std::atomic_store_sub(&t, val); };

    test_store_operations_pointer<T>(load, store, store_add, store_sub);
    ASSERT_NOEXCEPT(std::atomic_store_add(&t, 0));
    ASSERT_NOEXCEPT(std::atomic_store_sub(&t, 0));
  }
  {
    typedef std::atomic<T> A;
    volatile A t;

    auto load      = [&]() { return t.load(); };
    auto store     = [&](T val) { t.store(val); };
    auto store_add = [&](std::ptrdiff_t val, auto) { std::atomic_store_add(&t, val); };
    auto store_sub = [&](std::ptrdiff_t val, auto) { std::atomic_store_sub(&t, val); };

    test_store_operations_pointer<T>(load, store, store_add, store_sub);
    ASSERT_NOEXCEPT(std::atomic_store_add(&t, 0));
    ASSERT_NOEXCEPT(std::atomic_store_sub(&t, 0));
  }
}

int main(int, char**) {
  TestEachIntegralType<TestFn>()();
  testp<int*>();
  testp<const int*>();

  return 0;
}
