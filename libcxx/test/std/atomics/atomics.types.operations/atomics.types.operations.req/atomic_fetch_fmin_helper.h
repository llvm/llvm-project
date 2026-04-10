//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_ATOMICS_ATOMIC_FETCH_FMIN_HELPER_H
#define TEST_STD_ATOMICS_ATOMIC_FETCH_FMIN_HELPER_H

#include <cassert>
#include <cmath>
#include <limits>
#include <type_traits>

#include "test_macros.h"

// Test fetch_min for floating-point types
// NOTE: NaN handling and signed zero comparison (-0.0 vs +0.0) are unspecified
// LoadOp: () -> T - reads current value
// StoreOp: (T) -> void - stores a value
// MinOp: (T, memory_order) -> T - performs fetch_min operation
template <class T, class LoadOp, class StoreOp, class MinOp>
void test_fetch_fmin(LoadOp load, StoreOp store, MinOp min_op) {
  static_assert(std::is_floating_point_v<T>);

  constexpr T inf = std::numeric_limits<T>::infinity();

  // Test basic fetch_min (update to smaller value)
  {
    store(T(10.0));
    std::same_as<T> decltype(auto) old = min_op(T(5.0), std::memory_order_seq_cst);
    assert(old == T(10.0));
    assert(load() == T(5.0));
  }

  // Test with larger value (no change)
  {
    store(T(10.0));
    std::same_as<T> decltype(auto) old = min_op(T(20.0), std::memory_order_seq_cst);
    assert(old == T(10.0));
    assert(load() == T(10.0));
  }

  // Test with negative values
  {
    store(T(-5.0));
    std::same_as<T> decltype(auto) old = min_op(T(-10.0), std::memory_order_seq_cst);
    assert(old == T(-5.0));
    assert(load() == T(-10.0));
  }

  {
    store(T(-10.0));
    std::same_as<T> decltype(auto) old = min_op(T(-5.0), std::memory_order_seq_cst);
    assert(old == T(-10.0));
    assert(load() == T(-10.0));
  }

  // Test infinity
  {
    store(inf);
    std::same_as<T> decltype(auto) old = min_op(T(1.0), std::memory_order_seq_cst);
    assert(old == inf);
    assert(load() == T(1.0));
  }

  {
    store(T(1.0));
    std::same_as<T> decltype(auto) old = min_op(-inf, std::memory_order_seq_cst);
    assert(old == T(1.0));
    assert(load() == -inf);
  }

  // Test different memory orderings
  {
    store(T(15.0));
    std::same_as<T> decltype(auto) old = min_op(T(8.0), std::memory_order_relaxed);
    assert(old == T(15.0));
    assert(load() == T(8.0));
  }

  {
    store(T(10.0));
    std::same_as<T> decltype(auto) old = min_op(T(3.0), std::memory_order_acquire);
    assert(old == T(10.0));
    assert(load() == T(3.0));
  }

  {
    store(T(10.0));
    std::same_as<T> decltype(auto) old = min_op(T(2.0), std::memory_order_release);
    assert(old == T(10.0));
    assert(load() == T(2.0));
  }

  {
    store(T(10.0));
    std::same_as<T> decltype(auto) old = min_op(T(7.0), std::memory_order_acq_rel);
    assert(old == T(10.0));
    assert(load() == T(7.0));
  }
}

#endif // TEST_STD_ATOMICS_ATOMIC_FETCH_FMIN_HELPER_H
