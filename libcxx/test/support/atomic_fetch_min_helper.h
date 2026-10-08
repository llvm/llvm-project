//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_ATOMICS_ATOMIC_FETCH_MIN_HELPER_H
#define TEST_STD_ATOMICS_ATOMIC_FETCH_MIN_HELPER_H

#include <cassert>
#include <type_traits>

#include "test_macros.h"

// Test fetch_min for integral types
// LoadOp: () -> T - reads current value
// StoreOp: (T) -> void - stores a value
// MinOp: (T, memory_order) -> T - performs fetch_min operation
template <class T, class LoadOp, class StoreOp, class MinOp>
void test_fetch_min_integral(LoadOp load, StoreOp store, MinOp min) {
  // Test basic fetch_min (update to smaller value)
  {
    store(T(10));
    T old = min(T(5), std::memory_order_seq_cst);
    assert(old == T(10));
    assert(load() == T(5));
  }

  // Test with larger value (no change)
  {
    store(T(10));
    T old = min(T(20), std::memory_order_seq_cst);
    assert(old == T(10));
    assert(load() == T(10));
  }

  // Test different memory orderings
  {
    store(T(15));
    T old = min(T(8), std::memory_order_seq_cst);
    assert(old == T(15));
    assert(load() == T(8));
  }

  {
    store(T(10));
    T old = min(T(3), std::memory_order_relaxed);
    assert(old == T(10));
    assert(load() == T(3));
  }

  {
    store(T(10));
    T old = min(T(2), std::memory_order_acquire);
    assert(old == T(10));
    assert(load() == T(2));
  }

  {
    store(T(10));
    T old = min(T(7), std::memory_order_release);
    assert(old == T(10));
    assert(load() == T(7));
  }

  {
    store(T(10));
    T old = min(T(4), std::memory_order_acq_rel);
    assert(old == T(10));
    assert(load() == T(4));
  }

  {
    store(T(15));
    T old = min(T(6), std::memory_order_relaxed);
    assert(old == T(15));
    assert(load() == T(6));
  }

  // Test return type
  {
    store(T(10));
    static_assert(std::is_same_v<decltype(min(T(5), std::memory_order_seq_cst)), T>);
  }
}

// Test fetch_min for pointer types
// LoadOp: () -> T* - reads current pointer value
// StoreOp: (T*) -> void - stores a pointer value
// MinOp: (T*, memory_order) -> T* - performs fetch_min operation
template <class T, class LoadOp, class StoreOp, class MinOp>
void test_fetch_min_pointer(T* p0, T* p2, T* p4, LoadOp load, StoreOp store, MinOp min) {
  // Test basic fetch_min (update to smaller pointer)
  {
    store(p2);
    T* old = min(p0, std::memory_order_seq_cst);
    assert(old == p2);
    assert(load() == p0);
  }

  // Test with larger pointer (no change)
  {
    store(p2);
    T* old = min(p4, std::memory_order_seq_cst);
    assert(old == p2);
    assert(load() == p2);
  }

  // Test different memory orderings
  {
    store(p4);
    T* old = min(p2, std::memory_order_seq_cst);
    assert(old == p4);
    assert(load() == p2);
  }

  {
    store(p4);
    T* old = min(p2, std::memory_order_relaxed);
    assert(old == p4);
    assert(load() == p2);
  }

  {
    store(p4);
    T* old = min(p0, std::memory_order_acquire);
    assert(old == p4);
    assert(load() == p0);
  }

  {
    store(p4);
    T* old = min(p0, std::memory_order_release);
    assert(old == p4);
    assert(load() == p0);
  }

  // Test return type
  {
    store(p2);
    static_assert(std::is_same_v<decltype(min(p0, std::memory_order_seq_cst)), T*>);
  }
}

#endif // TEST_STD_ATOMICS_ATOMIC_FETCH_MIN_HELPER_H
