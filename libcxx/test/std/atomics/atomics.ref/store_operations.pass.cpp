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

// Test atomic_ref<T> member functions:
//   store_add, store_sub, store_and, store_or, store_xor
//
// void store_add(T operand, memory_order order = memory_order::seq_cst) const noexcept;
// (Same pattern for store_sub, store_and, store_or, store_xor)

#include <atomic>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "../atomics.types.operations/atomics.types.operations.req/atomic_store_operations_helper.h"

template <class T>
struct TestFn {
  void operator()() const {
    {
      T val;
      std::atomic_ref<T> ref(val);

      auto load      = [&]() { return ref.load(); };
      auto store     = [&](T v) { ref.store(v); };
      auto store_add = [&](T v, auto order) { ref.store_add(v, order); };
      auto store_sub = [&](T v, auto order) { ref.store_sub(v, order); };
      auto store_and = [&](T v, auto order) { ref.store_and(v, order); };
      auto store_or  = [&](T v, auto order) { ref.store_or(v, order); };
      auto store_xor = [&](T v, auto order) { ref.store_xor(v, order); };

      test_store_operations_integral<T>(load, store, store_add, store_sub, store_and, store_or, store_xor);
      ASSERT_NOEXCEPT(ref.store_add(T(0)));
      ASSERT_NOEXCEPT(ref.store_sub(T(0)));
      ASSERT_NOEXCEPT(ref.store_and(T(0)));
      ASSERT_NOEXCEPT(ref.store_or(T(0)));
      ASSERT_NOEXCEPT(ref.store_xor(T(0)));
    }
  }
};

template <class T>
void testp() {
  {
    T val;
    std::atomic_ref<T> ref(val);

    auto load      = [&]() { return ref.load(); };
    auto store     = [&](T v) { ref.store(v); };
    auto store_add = [&](std::ptrdiff_t v, auto order) { ref.store_add(v, order); };
    auto store_sub = [&](std::ptrdiff_t v, auto order) { ref.store_sub(v, order); };

    test_store_operations_pointer<T>(load, store, store_add, store_sub);
    ASSERT_NOEXCEPT(ref.store_add(0));
    ASSERT_NOEXCEPT(ref.store_sub(0));
  }
}

template <class T>
void testf() {
  {
    T val;
    std::atomic_ref<T> ref(val);

    auto load      = [&]() { return ref.load(); };
    auto store     = [&](T v) { ref.store(v); };
    auto store_add = [&](T v, auto order) { ref.store_add(v, order); };
    auto store_sub = [&](T v, auto order) { ref.store_sub(v, order); };

    test_store_operations_floating<T>(load, store, store_add, store_sub);
    ASSERT_NOEXCEPT(ref.store_add(T(0)));
    ASSERT_NOEXCEPT(ref.store_sub(T(0)));
  }
}

int main(int, char**) {
  TestEachIntegralType<TestFn>()();
  testp<int*>();
  testp<const int*>();
  testf<float>();
  testf<double>();

  return 0;
}
