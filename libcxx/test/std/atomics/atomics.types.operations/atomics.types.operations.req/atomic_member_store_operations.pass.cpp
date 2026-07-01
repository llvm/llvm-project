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

// Test atomic<T> member functions:
//   store_add, store_sub, store_and, store_or, store_xor
//
// void store_add(T operand, memory_order order = memory_order::seq_cst) volatile noexcept;
// void store_add(T operand, memory_order order = memory_order::seq_cst) noexcept;
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
    // Test non-volatile atomic<T>
    {
      typedef std::atomic<T> A;
      A t;

      auto load      = [&]() { return t.load(); };
      auto store     = [&](T val) { t.store(val); };
      auto store_add = [&](T val, auto order) { t.store_add(val, order); };
      auto store_sub = [&](T val, auto order) { t.store_sub(val, order); };
      auto store_and = [&](T val, auto order) { t.store_and(val, order); };
      auto store_or  = [&](T val, auto order) { t.store_or(val, order); };
      auto store_xor = [&](T val, auto order) { t.store_xor(val, order); };

      test_store_operations_integral<T>(load, store, store_add, store_sub, store_and, store_or, store_xor);
      ASSERT_NOEXCEPT(t.store_add(T(0)));
      ASSERT_NOEXCEPT(t.store_sub(T(0)));
      ASSERT_NOEXCEPT(t.store_and(T(0)));
      ASSERT_NOEXCEPT(t.store_or(T(0)));
      ASSERT_NOEXCEPT(t.store_xor(T(0)));
    }

    // Test volatile atomic<T>
    {
      typedef std::atomic<T> A;
      volatile A t;

      auto load      = [&]() { return t.load(); };
      auto store     = [&](T val) { t.store(val); };
      auto store_add = [&](T val, auto order) { t.store_add(val, order); };
      auto store_sub = [&](T val, auto order) { t.store_sub(val, order); };
      auto store_and = [&](T val, auto order) { t.store_and(val, order); };
      auto store_or  = [&](T val, auto order) { t.store_or(val, order); };
      auto store_xor = [&](T val, auto order) { t.store_xor(val, order); };

      test_store_operations_integral<T>(load, store, store_add, store_sub, store_and, store_or, store_xor);
      ASSERT_NOEXCEPT(t.store_add(T(0)));
      ASSERT_NOEXCEPT(t.store_sub(T(0)));
      ASSERT_NOEXCEPT(t.store_and(T(0)));
      ASSERT_NOEXCEPT(t.store_or(T(0)));
      ASSERT_NOEXCEPT(t.store_xor(T(0)));
    }
  }
};

template <class T>
void testp() {
  // Test non-volatile atomic<T*>
  {
    typedef std::atomic<T> A;
    A t;

    auto load      = [&]() { return t.load(); };
    auto store     = [&](T val) { t.store(val); };
    auto store_add = [&](std::ptrdiff_t val, auto order) { t.store_add(val, order); };
    auto store_sub = [&](std::ptrdiff_t val, auto order) { t.store_sub(val, order); };

    test_store_operations_pointer<T>(load, store, store_add, store_sub);
    ASSERT_NOEXCEPT(t.store_add(0));
    ASSERT_NOEXCEPT(t.store_sub(0));
  }

  // Test volatile atomic<T*>
  {
    typedef std::atomic<T> A;
    volatile A t;

    auto load      = [&]() { return t.load(); };
    auto store     = [&](T val) { t.store(val); };
    auto store_add = [&](std::ptrdiff_t val, auto order) { t.store_add(val, order); };
    auto store_sub = [&](std::ptrdiff_t val, auto order) { t.store_sub(val, order); };

    test_store_operations_pointer<T>(load, store, store_add, store_sub);
    ASSERT_NOEXCEPT(t.store_add(0));
    ASSERT_NOEXCEPT(t.store_sub(0));
  }
}

int main(int, char**) {
  TestEachIntegralType<TestFn>()();
  testp<int*>();
  testp<const int*>();

  return 0;
}
