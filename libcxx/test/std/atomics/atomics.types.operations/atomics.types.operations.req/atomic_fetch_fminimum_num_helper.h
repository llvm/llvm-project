//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_ATOMICS_ATOMIC_FETCH_FMINIMUM_NUM_HELPER_H
#define TEST_STD_ATOMICS_ATOMIC_FETCH_FMINIMUM_NUM_HELPER_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "test_macros.h"

// Create a NaN with a specific payload to get different bit representations
template <class T>
T make_nan_with_payload(unsigned int payload) {
  static_assert(std::is_floating_point_v<T>);
  T nan = std::numeric_limits<T>::quiet_NaN();
  // Create NaN with different bit pattern by manipulating the payload
  using UIntType = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
  UIntType bits;
  std::memcpy(&bits, &nan, sizeof(T));
  // Modify the mantissa bits (payload) while keeping it a NaN
  bits = (bits & ~UIntType(0xFFFFF)) | (payload & 0xFFFFF);
  std::memcpy(&nan, &bits, sizeof(T));
  return nan;
}

// Test fetch_fminimum_num for floating-point types
// LoadOp: () -> T - reads current value
// StoreOp: (T) -> void - stores a value
// FMinimumNumOp: (T, memory_order) -> T - performs fetch_fminimum_num operation
template <class T, class LoadOp, class StoreOp, class FMinimumNumOp>
void test_fetch_fminimum_num(LoadOp load, StoreOp store, FMinimumNumOp fminimum_num) {
  static_assert(std::is_floating_point_v<T>);

  constexpr T nan = std::numeric_limits<T>::quiet_NaN();
  constexpr T inf = std::numeric_limits<T>::infinity();

  // Test basic fetch_fminimum_num (update to smaller value)
  {
    store(T(10.0));
    T old = fminimum_num(T(5.0), std::memory_order_seq_cst);
    assert(old == T(10.0));
    assert(load() == T(5.0));
  }

  // Test with larger value (no change)
  {
    store(T(10.0));
    T old = fminimum_num(T(20.0), std::memory_order_seq_cst);
    assert(old == T(10.0));
    assert(load() == T(10.0));
  }

  // Test with negative values
  {
    store(T(-5.0));
    T old = fminimum_num(T(-10.0), std::memory_order_seq_cst);
    assert(old == T(-5.0));
    assert(load() == T(-10.0));
  }

  {
    store(T(-10.0));
    T old = fminimum_num(T(-5.0), std::memory_order_seq_cst);
    assert(old == T(-10.0));
    assert(load() == T(-10.0));
  }

  // Test NaN handling: favor non-NaN values
  {
    store(nan);
    T old = fminimum_num(T(5.0), std::memory_order_seq_cst);
    assert(std::isnan(old));
    assert(load() == T(5.0));
  }

  {
    store(T(5.0));
    T old = fminimum_num(nan, std::memory_order_seq_cst);
    assert(old == T(5.0));
    assert(load() == T(5.0));
  }

  // Both NaN: return NaN
  {
    store(nan);
    T old = fminimum_num(nan, std::memory_order_seq_cst);
    assert(std::isnan(old));
    assert(std::isnan(load()));
  }

  // Test signed zero handling: -0.0 < +0.0
  {
    store(T(+0.0));
    T old = fminimum_num(T(-0.0), std::memory_order_seq_cst);
    assert(old == T(+0.0));
    assert(!std::signbit(old));
    assert(std::signbit(load()));
  }

  {
    store(T(-0.0));
    T old = fminimum_num(T(+0.0), std::memory_order_seq_cst);
    assert(old == T(-0.0));
    assert(std::signbit(old));
    assert(std::signbit(load()));
  }

  // Test infinity
  {
    store(inf);
    T old = fminimum_num(T(1.0), std::memory_order_seq_cst);
    assert(old == inf);
    assert(load() == T(1.0));
  }

  {
    store(T(1.0));
    T old = fminimum_num(-inf, std::memory_order_seq_cst);
    assert(old == T(1.0));
    assert(load() == -inf);
  }

  // Test different memory orderings
  {
    store(T(15.0));
    T old = fminimum_num(T(8.0), std::memory_order_relaxed);
    assert(old == T(15.0));
    assert(load() == T(8.0));
  }

  {
    store(T(10.0));
    T old = fminimum_num(T(3.0), std::memory_order_acquire);
    assert(old == T(10.0));
    assert(load() == T(3.0));
  }

  {
    store(T(10.0));
    T old = fminimum_num(T(2.0), std::memory_order_release);
    assert(old == T(10.0));
    assert(load() == T(2.0));
  }

  {
    store(T(10.0));
    T old = fminimum_num(T(7.0), std::memory_order_acq_rel);
    assert(old == T(10.0));
    assert(load() == T(7.0));
  }

  // Test with different NaN representations
  {
    T nan1 = make_nan_with_payload<T>(1);

    // Store NaN with payload 1, fetch with non-NaN
    store(nan1);
    T old = fminimum_num(T(5.0), std::memory_order_seq_cst);
    assert(std::isnan(old));
    assert(load() == T(5.0));
  }

  {
    T nan2 = make_nan_with_payload<T>(2);

    // Store non-NaN, fetch with NaN with payload 2
    store(T(5.0));
    T old = fminimum_num(nan2, std::memory_order_seq_cst);
    assert(old == T(5.0));
    assert(load() == T(5.0));
  }

  {
    T nan1 = make_nan_with_payload<T>(1);
    T nan2 = make_nan_with_payload<T>(2);

    // Store NaN with payload 1, fetch with NaN with payload 2
    store(nan1);
    T old = fminimum_num(nan2, std::memory_order_seq_cst);
    assert(std::isnan(old));
    assert(std::isnan(load()));
    // Result should be one of the NaN values (implementation-defined which)
  }

  // Test return type
  {
    store(T(10.0));
    static_assert(std::is_same_v<decltype(fminimum_num(T(5.0), std::memory_order_seq_cst)), T>);
  }
}

#endif // TEST_STD_ATOMICS_ATOMIC_FETCH_FMINIMUM_NUM_HELPER_H
