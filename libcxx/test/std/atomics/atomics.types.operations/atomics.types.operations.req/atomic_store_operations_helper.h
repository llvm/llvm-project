//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_ATOMICS_ATOMIC_STORE_OPERATIONS_HELPER_H
#define TEST_STD_ATOMICS_ATOMIC_STORE_OPERATIONS_HELPER_H

#include <cassert>
#include <type_traits>

#include "test_macros.h"

// Test store_* operations for integral types
// LoadOp: () -> T - reads current value
// StoreOp: (T) -> void - stores a value
// StoreAddOp: (T, memory_order) -> void - performs store_add operation
// StoreSubOp: (T, memory_order) -> void - performs store_sub operation
// StoreAndOp: (T, memory_order) -> void - performs store_and operation
// StoreOrOp: (T, memory_order) -> void - performs store_or operation
// StoreXorOp: (T, memory_order) -> void - performs store_xor operation
template <class T,
          class LoadOp,
          class StoreOp,
          class StoreAddOp,
          class StoreSubOp,
          class StoreAndOp,
          class StoreOrOp,
          class StoreXorOp>
void test_store_operations_integral(
    LoadOp load,
    StoreOp store,
    StoreAddOp store_add,
    StoreSubOp store_sub,
    StoreAndOp store_and,
    StoreOrOp store_or,
    StoreXorOp store_xor) {
  static_assert(std::is_integral_v<T>);

  // Test store_add
  {
    store(T(1));
    store_add(T(2), std::memory_order_seq_cst);
    assert(load() == T(3));
  }

  {
    store(T(10));
    store_add(T(0), std::memory_order_seq_cst);
    assert(load() == T(10));
  }

  if constexpr (std::is_signed_v<T>) {
    store(T(5));
    store_add(T(-3), std::memory_order_seq_cst);
    assert(load() == T(2));

    store(T(-10));
    store_add(T(15), std::memory_order_seq_cst);
    assert(load() == T(5));
  }

  {
    store(T(10));
    store_add(T(5), std::memory_order_relaxed);
    assert(load() == T(15));
  }

  {
    store(T(20));
    store_add(T(10), std::memory_order_release);
    assert(load() == T(30));
  }

  // Test store_sub
  {
    store(T(10));
    store_sub(T(3), std::memory_order_seq_cst);
    assert(load() == T(7));
  }

  {
    store(T(10));
    store_sub(T(0), std::memory_order_seq_cst);
    assert(load() == T(10));
  }

  if constexpr (std::is_signed_v<T>) {
    store(T(5));
    store_sub(T(-3), std::memory_order_seq_cst);
    assert(load() == T(8));

    store(T(-10));
    store_sub(T(5), std::memory_order_seq_cst);
    assert(load() == T(-15));
  }

  {
    store(T(20));
    store_sub(T(5), std::memory_order_relaxed);
    assert(load() == T(15));
  }

  {
    store(T(50));
    store_sub(T(10), std::memory_order_release);
    assert(load() == T(40));
  }

  // Test store_and
  {
    store(T(0b1111));
    store_and(T(0b1010), std::memory_order_seq_cst);
    assert(load() == T(0b1010));
  }

  {
    store(T(0b1111));
    store_and(T(0b0000), std::memory_order_seq_cst);
    assert(load() == T(0b0000));
  }

  {
    store(T(0b1111));
    store_and(T(0b1111), std::memory_order_seq_cst);
    assert(load() == T(0b1111));
  }

  {
    store(T(0b1100));
    store_and(T(0b1010), std::memory_order_relaxed);
    assert(load() == T(0b1000));
  }

  {
    store(T(0b1111));
    store_and(T(0b0101), std::memory_order_release);
    assert(load() == T(0b0101));
  }

  // Test store_or
  {
    store(T(0b1010));
    store_or(T(0b0101), std::memory_order_seq_cst);
    assert(load() == T(0b1111));
  }

  {
    store(T(0b1010));
    store_or(T(0b0000), std::memory_order_seq_cst);
    assert(load() == T(0b1010));
  }

  {
    store(T(0b0000));
    store_or(T(0b1111), std::memory_order_seq_cst);
    assert(load() == T(0b1111));
  }

  {
    store(T(0b1000));
    store_or(T(0b0010), std::memory_order_relaxed);
    assert(load() == T(0b1010));
  }

  {
    store(T(0b0101));
    store_or(T(0b1010), std::memory_order_release);
    assert(load() == T(0b1111));
  }

  // Test store_xor
  {
    store(T(0b1100));
    store_xor(T(0b1010), std::memory_order_seq_cst);
    assert(load() == T(0b0110));
  }

  {
    store(T(0b1010));
    store_xor(T(0b0000), std::memory_order_seq_cst);
    assert(load() == T(0b1010));
  }

  {
    store(T(0b1111));
    store_xor(T(0b1111), std::memory_order_seq_cst);
    assert(load() == T(0b0000));
  }

  {
    store(T(0b1100));
    store_xor(T(0b0110), std::memory_order_relaxed);
    assert(load() == T(0b1010));
  }

  {
    store(T(0b1010));
    store_xor(T(0b1111), std::memory_order_release);
    assert(load() == T(0b0101));
  }

  // Test return types (should all be void)
  {
    store(T(1));
    static_assert(std::is_same_v<decltype(store_add(T(2), std::memory_order_seq_cst)), void>);
    static_assert(std::is_same_v<decltype(store_sub(T(2), std::memory_order_seq_cst)), void>);
    static_assert(std::is_same_v<decltype(store_and(T(2), std::memory_order_seq_cst)), void>);
    static_assert(std::is_same_v<decltype(store_or(T(2), std::memory_order_seq_cst)), void>);
    static_assert(std::is_same_v<decltype(store_xor(T(2), std::memory_order_seq_cst)), void>);
  }
}

// Test store_add and store_sub for floating-point types
// LoadOp: () -> T - reads current value
// StoreOp: (T) -> void - stores a value
// StoreAddOp: (T, memory_order) -> void - performs store_add operation
// StoreSubOp: (T, memory_order) -> void - performs store_sub operation
template <class T, class LoadOp, class StoreOp, class StoreAddOp, class StoreSubOp>
  requires std::is_floating_point_v<T>
void test_store_operations_floating(LoadOp load, StoreOp store, StoreAddOp store_add, StoreSubOp store_sub) {
  // Test store_add
  {
    store(T(1.5));
    store_add(T(2.5), std::memory_order_seq_cst);
    assert(load() == T(4.0));
  }

  {
    store(T(10.5));
    store_add(T(0.0), std::memory_order_seq_cst);
    assert(load() == T(10.5));
  }

  {
    store(T(5.5));
    store_add(T(-3.5), std::memory_order_seq_cst);
    assert(load() == T(2.0));
  }

  {
    store(T(-10.0));
    store_add(T(15.5), std::memory_order_seq_cst);
    assert(load() == T(5.5));
  }

  {
    store(T(10.0));
    store_add(T(5.5), std::memory_order_relaxed);
    assert(load() == T(15.5));
  }

  {
    store(T(20.0));
    store_add(T(10.5), std::memory_order_release);
    assert(load() == T(30.5));
  }

  // Test store_sub
  {
    store(T(10.0));
    store_sub(T(3.5), std::memory_order_seq_cst);
    assert(load() == T(6.5));
  }

  {
    store(T(10.5));
    store_sub(T(0.0), std::memory_order_seq_cst);
    assert(load() == T(10.5));
  }

  {
    store(T(5.5));
    store_sub(T(-3.5), std::memory_order_seq_cst);
    assert(load() == T(9.0));
  }

  {
    store(T(-10.0));
    store_sub(T(5.5), std::memory_order_seq_cst);
    assert(load() == T(-15.5));
  }

  {
    store(T(20.0));
    store_sub(T(5.5), std::memory_order_relaxed);
    assert(load() == T(14.5));
  }

  {
    store(T(50.0));
    store_sub(T(10.5), std::memory_order_release);
    assert(load() == T(39.5));
  }

  // Test return types (should be void)
  {
    store(T(1.0));
    static_assert(std::is_same_v<decltype(store_add(T(2.0), std::memory_order_seq_cst)), void>);
    static_assert(std::is_same_v<decltype(store_sub(T(2.0), std::memory_order_seq_cst)), void>);
  }
}

// Test store_add and store_sub for pointer types
// LoadOp: () -> T* - reads current value
// StoreOp: (T*) -> void - stores a value
// StoreAddOp: (ptrdiff_t, memory_order) -> void - performs store_add operation
// StoreSubOp: (ptrdiff_t, memory_order) -> void - performs store_sub operation
template <class T, class LoadOp, class StoreOp, class StoreAddOp, class StoreSubOp>
  requires std::is_pointer_v<T>
void test_store_operations_pointer(LoadOp load, StoreOp store, StoreAddOp store_add, StoreSubOp store_sub) {
  using X   = std::remove_pointer_t<T>;
  X arr[10] = {};

  // Test store_add
  {
    store(&arr[0]);
    store_add(3, std::memory_order_seq_cst);
    assert(load() == &arr[3]);
  }

  {
    store(&arr[5]);
    store_add(0, std::memory_order_seq_cst);
    assert(load() == &arr[5]);
  }

  {
    store(&arr[7]);
    store_add(-3, std::memory_order_seq_cst);
    assert(load() == &arr[4]);
  }

  {
    store(&arr[0]);
    store_add(2, std::memory_order_relaxed);
    assert(load() == &arr[2]);
  }

  {
    store(&arr[1]);
    store_add(4, std::memory_order_release);
    assert(load() == &arr[5]);
  }

  // Test store_sub
  {
    store(&arr[7]);
    store_sub(3, std::memory_order_seq_cst);
    assert(load() == &arr[4]);
  }

  {
    store(&arr[5]);
    store_sub(0, std::memory_order_seq_cst);
    assert(load() == &arr[5]);
  }

  {
    store(&arr[3]);
    store_sub(-4, std::memory_order_seq_cst);
    assert(load() == &arr[7]);
  }

  {
    store(&arr[9]);
    store_sub(2, std::memory_order_relaxed);
    assert(load() == &arr[7]);
  }

  {
    store(&arr[8]);
    store_sub(4, std::memory_order_release);
    assert(load() == &arr[4]);
  }

  // Test return types (should be void)
  {
    store(&arr[0]);
    static_assert(std::is_same_v<decltype(store_add(1, std::memory_order_seq_cst)), void>);
    static_assert(std::is_same_v<decltype(store_sub(1, std::memory_order_seq_cst)), void>);
  }
}

#endif // TEST_STD_ATOMICS_ATOMIC_STORE_OPERATIONS_HELPER_H
