//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_ATOMICS_ATOMIC_FETCH_MAX_HELPER_H
#define TEST_STD_ATOMICS_ATOMIC_FETCH_MAX_HELPER_H

#include <cassert>
#include <type_traits>

#include "test_macros.h"

// Test fetch_max for integral types
// LoadOp: () -> T - reads current value
// StoreOp: (T) -> void - stores a value
// MaxOp: (T, memory_order) -> T - performs fetch_max operation
template <class T, class LoadOp, class StoreOp, class MaxOp>
void test_fetch_max_integral(LoadOp load, StoreOp store, MaxOp max) {
  // Test basic fetch_max (update to larger value)
  {
    store(T(10));
    T old = max(T(20), std::memory_order_seq_cst);
    assert(old == T(10));
    assert(load() == T(20));
  }

  // Test with smaller value (no change)
  {
    store(T(10));
    T old = max(T(5), std::memory_order_seq_cst);
    assert(old == T(10));
    assert(load() == T(10));
  }

  // Test different memory orderings
  {
    store(T(15));
    T old = max(T(25), std::memory_order_seq_cst);
    assert(old == T(15));
    assert(load() == T(25));
  }

  {
    store(T(10));
    T old = max(T(25), std::memory_order_relaxed);
    assert(old == T(10));
    assert(load() == T(25));
  }

  {
    store(T(10));
    T old = max(T(30), std::memory_order_acquire);
    assert(old == T(10));
    assert(load() == T(30));
  }

  {
    store(T(10));
    T old = max(T(15), std::memory_order_release);
    assert(old == T(10));
    assert(load() == T(15));
  }

  {
    store(T(10));
    T old = max(T(22), std::memory_order_acq_rel);
    assert(old == T(10));
    assert(load() == T(22));
  }

  {
    store(T(15));
    T old = max(T(35), std::memory_order_relaxed);
    assert(old == T(15));
    assert(load() == T(35));
  }

  // Test return type
  {
    store(T(10));
    static_assert(std::is_same_v<decltype(max(T(20), std::memory_order_seq_cst)), T>);
  }
}

// Test fetch_max for pointer types
// LoadOp: () -> T* - reads current pointer value
// StoreOp: (T*) -> void - stores a pointer value
// MaxOp: (T*, memory_order) -> T* - performs fetch_max operation
template <class T, class LoadOp, class StoreOp, class MaxOp>
void test_fetch_max_pointer(T* p0, T* p2, T* p4, LoadOp load, StoreOp store, MaxOp max) {
  // Test basic fetch_max (update to larger pointer)
  {
    store(p2);
    T* old = max(p4, std::memory_order_seq_cst);
    assert(old == p2);
    assert(load() == p4);
  }

  // Test with smaller pointer (no change)
  {
    store(p2);
    T* old = max(p0, std::memory_order_seq_cst);
    assert(old == p2);
    assert(load() == p2);
  }

  // Test different memory orderings
  {
    store(p0);
    T* old = max(p2, std::memory_order_seq_cst);
    assert(old == p0);
    assert(load() == p2);
  }

  {
    store(p0);
    T* old = max(p2, std::memory_order_relaxed);
    assert(old == p0);
    assert(load() == p2);
  }

  {
    store(p0);
    T* old = max(p4, std::memory_order_acquire);
    assert(old == p0);
    assert(load() == p4);
  }

  {
    store(p0);
    T* old = max(p4, std::memory_order_release);
    assert(old == p0);
    assert(load() == p4);
  }

  // Test return type
  {
    store(p2);
    static_assert(std::is_same_v<decltype(max(p4, std::memory_order_seq_cst)), T*>);
  }
}

#endif // TEST_STD_ATOMICS_ATOMIC_FETCH_MAX_HELPER_H
