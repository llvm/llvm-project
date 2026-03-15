//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// XFAIL: !has-64-bit-atomics

// template<class T>
// T atomic_fetch_max_explicit(volatile atomic<T>*, typename atomic<T>::value_type,
//                              memory_order) noexcept;
//
// template<class T>
// T atomic_fetch_max_explicit(atomic<T>*, typename atomic<T>::value_type,
//                              memory_order) noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "atomic_fetch_max_helper.h"

template <class T>
struct TestFn {
  void operator()() const {
    // Test non-volatile
    {
      std::atomic<T> a;
      auto load  = [&]() { return a.load(); };
      auto store = [&](T val) { a.store(val); };
      auto max   = [&](T val, auto order) { return std::atomic_fetch_max_explicit(&a, val, order); };
      ASSERT_NOEXCEPT(std::atomic_fetch_max_explicit(&a, T(0), std::memory_order_seq_cst));
      test_fetch_max_integral<T>(load, store, max);
    }

    // Test volatile
    {
      volatile std::atomic<T> a;
      auto load  = [&]() { return a.load(); };
      auto store = [&](T val) { a.store(val); };
      auto max   = [&](T val, auto order) { return std::atomic_fetch_max_explicit(&a, val, order); };
      ASSERT_NOEXCEPT(std::atomic_fetch_max_explicit(&a, T(0), std::memory_order_seq_cst));
      test_fetch_max_integral<T>(load, store, max);
    }
  }
};

template <class T>
void test_pointer() {
  T arr[5];
  T* p0 = &arr[0];
  T* p2 = &arr[2];
  T* p4 = &arr[4];

  // Test non-volatile
  {
    std::atomic<T*> a;
    auto load  = [&]() { return a.load(); };
    auto store = [&](T* val) { a.store(val); };
    auto max   = [&](T* val, auto order) { return std::atomic_fetch_max_explicit(&a, val, order); };
    ASSERT_NOEXCEPT(std::atomic_fetch_max_explicit(&a, p0, std::memory_order_seq_cst));
    test_fetch_max_pointer<T>(p0, p2, p4, load, store, max);
  }

  // Test volatile
  {
    volatile std::atomic<T*> a;
    auto load  = [&]() { return a.load(); };
    auto store = [&](T* val) { a.store(val); };
    auto max   = [&](T* val, auto order) { return std::atomic_fetch_max_explicit(&a, val, order); };
    ASSERT_NOEXCEPT(std::atomic_fetch_max_explicit(&a, p0, std::memory_order_seq_cst));
    test_fetch_max_pointer<T>(p0, p2, p4, load, store, max);
  }
}

int main(int, char**) {
  TestEachIntegralType<TestFn>()();
  test_pointer<int>();
  test_pointer<char>();

  return 0;
}
