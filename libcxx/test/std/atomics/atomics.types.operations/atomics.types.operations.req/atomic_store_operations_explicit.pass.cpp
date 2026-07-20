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

// Test all atomic_store_*_explicit non-member functions:
//   atomic_store_add_explicit, atomic_store_sub_explicit,
//   atomic_store_and_explicit, atomic_store_or_explicit, atomic_store_xor_explicit
//
// template<class T>
//     void atomic_store_add_explicit(volatile atomic<T>*, atomic<T>::difference_type, memory_order) noexcept;
// template<class T>
//     void atomic_store_add_explicit(atomic<T>*, atomic<T>::difference_type, memory_order) noexcept;
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
      auto store_add = [&](T val, auto order) { std::atomic_store_add_explicit(&t, val, order); };
      auto store_sub = [&](T val, auto order) { std::atomic_store_sub_explicit(&t, val, order); };
      auto store_and = [&](T val, auto order) { std::atomic_store_and_explicit(&t, val, order); };
      auto store_or  = [&](T val, auto order) { std::atomic_store_or_explicit(&t, val, order); };
      auto store_xor = [&](T val, auto order) { std::atomic_store_xor_explicit(&t, val, order); };

      test_store_operations_integral<T>(load, store, store_add, store_sub, store_and, store_or, store_xor);
      ASSERT_NOEXCEPT(std::atomic_store_add_explicit(&t, T(0), std::memory_order_seq_cst));
      ASSERT_NOEXCEPT(std::atomic_store_sub_explicit(&t, T(0), std::memory_order_seq_cst));
      ASSERT_NOEXCEPT(std::atomic_store_and_explicit(&t, T(0), std::memory_order_seq_cst));
      ASSERT_NOEXCEPT(std::atomic_store_or_explicit(&t, T(0), std::memory_order_seq_cst));
      ASSERT_NOEXCEPT(std::atomic_store_xor_explicit(&t, T(0), std::memory_order_seq_cst));
    }
    {
      typedef std::atomic<T> A;
      volatile A t;

      auto load      = [&]() { return t.load(); };
      auto store     = [&](T val) { t.store(val); };
      auto store_add = [&](T val, auto order) { std::atomic_store_add_explicit(&t, val, order); };
      auto store_sub = [&](T val, auto order) { std::atomic_store_sub_explicit(&t, val, order); };
      auto store_and = [&](T val, auto order) { std::atomic_store_and_explicit(&t, val, order); };
      auto store_or  = [&](T val, auto order) { std::atomic_store_or_explicit(&t, val, order); };
      auto store_xor = [&](T val, auto order) { std::atomic_store_xor_explicit(&t, val, order); };

      test_store_operations_integral<T>(load, store, store_add, store_sub, store_and, store_or, store_xor);
      ASSERT_NOEXCEPT(std::atomic_store_add_explicit(&t, T(0), std::memory_order_seq_cst));
      ASSERT_NOEXCEPT(std::atomic_store_sub_explicit(&t, T(0), std::memory_order_seq_cst));
      ASSERT_NOEXCEPT(std::atomic_store_and_explicit(&t, T(0), std::memory_order_seq_cst));
      ASSERT_NOEXCEPT(std::atomic_store_or_explicit(&t, T(0), std::memory_order_seq_cst));
      ASSERT_NOEXCEPT(std::atomic_store_xor_explicit(&t, T(0), std::memory_order_seq_cst));
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
    auto store_add = [&](std::ptrdiff_t val, auto order) { std::atomic_store_add_explicit(&t, val, order); };
    auto store_sub = [&](std::ptrdiff_t val, auto order) { std::atomic_store_sub_explicit(&t, val, order); };

    test_store_operations_pointer<T>(load, store, store_add, store_sub);
    ASSERT_NOEXCEPT(std::atomic_store_add_explicit(&t, 0, std::memory_order_seq_cst));
    ASSERT_NOEXCEPT(std::atomic_store_sub_explicit(&t, 0, std::memory_order_seq_cst));
  }
  {
    typedef std::atomic<T> A;
    volatile A t;

    auto load      = [&]() { return t.load(); };
    auto store     = [&](T val) { t.store(val); };
    auto store_add = [&](std::ptrdiff_t val, auto order) { std::atomic_store_add_explicit(&t, val, order); };
    auto store_sub = [&](std::ptrdiff_t val, auto order) { std::atomic_store_sub_explicit(&t, val, order); };

    test_store_operations_pointer<T>(load, store, store_add, store_sub);
    ASSERT_NOEXCEPT(std::atomic_store_add_explicit(&t, 0, std::memory_order_seq_cst));
    ASSERT_NOEXCEPT(std::atomic_store_sub_explicit(&t, 0, std::memory_order_seq_cst));
  }
}

int main(int, char**) {
  TestEachIntegralType<TestFn>()();
  testp<int*>();
  testp<const int*>();

  return 0;
}
