//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_ATOMICS_ATOMIC_FETCH_FMAX_HELPER_H
#define TEST_STD_ATOMICS_ATOMIC_FETCH_FMAX_HELPER_H

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#include "test_macros.h"

// Test fetch_max for floating-point types
// NOTE: NaN handling and signed zero comparison (-0.0 vs +0.0) are unspecified
// LoadOp: () -> T - reads current value
// StoreOp: (T) -> void - stores a value
// MaxOp: (T, memory_order) -> T - performs fetch_max operation
template <class T, class LoadOp, class StoreOp, class MaxOp>
void test_fetch_fmax(LoadOp load, StoreOp store, MaxOp max_op) {
  static_assert(std::is_floating_point_v<T>);

  constexpr T inf = std::numeric_limits<T>::infinity();

  // Test basic fetch_max (update to larger value)
  {
    store(T(10.0));
    std::same_as<T> decltype(auto) old = max_op(T(20.0), std::memory_order_seq_cst);
    assert(old == T(10.0));
    assert(load() == T(20.0));
  }

  // Test with smaller value (no change)
  {
    store(T(10.0));
    std::same_as<T> decltype(auto) old = max_op(T(5.0), std::memory_order_seq_cst);
    assert(old == T(10.0));
    assert(load() == T(10.0));
  }

  // Test with negative values
  {
    store(T(-10.0));
    std::same_as<T> decltype(auto) old = max_op(T(-5.0), std::memory_order_seq_cst);
    assert(old == T(-10.0));
    assert(load() == T(-5.0));
  }

  {
    store(T(-5.0));
    std::same_as<T> decltype(auto) old = max_op(T(-10.0), std::memory_order_seq_cst);
    assert(old == T(-5.0));
    assert(load() == T(-5.0));
  }

  // Test infinity
  {
    store(T(1.0));
    std::same_as<T> decltype(auto) old = max_op(inf, std::memory_order_seq_cst);
    assert(old == T(1.0));
    assert(load() == inf);
  }

  {
    store(-inf);
    std::same_as<T> decltype(auto) old = max_op(T(1.0), std::memory_order_seq_cst);
    assert(old == -inf);
    assert(load() == T(1.0));
  }

  // Test different memory orderings
  {
    store(T(8.0));
    std::same_as<T> decltype(auto) old = max_op(T(15.0), std::memory_order_relaxed);
    assert(old == T(8.0));
    assert(load() == T(15.0));
  }

  {
    store(T(3.0));
    std::same_as<T> decltype(auto) old = max_op(T(10.0), std::memory_order_acquire);
    assert(old == T(3.0));
    assert(load() == T(10.0));
  }

  {
    store(T(2.0));
    std::same_as<T> decltype(auto) old = max_op(T(10.0), std::memory_order_release);
    assert(old == T(2.0));
    assert(load() == T(10.0));
  }

  {
    store(T(7.0));
    std::same_as<T> decltype(auto) old = max_op(T(10.0), std::memory_order_acq_rel);
    assert(old == T(7.0));
    assert(load() == T(10.0));
  }
}

#endif // TEST_STD_ATOMICS_ATOMIC_FETCH_FMAX_HELPER_H
