//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_ATOMICS_ATOMIC_FETCH_FMAXIMUM_HELPER_H
#define TEST_STD_ATOMICS_ATOMIC_FETCH_FMAXIMUM_HELPER_H

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

  // Use a byte array sized to T to handle all floating-point types
  alignas(T) unsigned char bits[sizeof(T)];
  std::memcpy(bits, &nan, sizeof(T));

  // Modify the mantissa bits (payload) while keeping it a NaN
  // For float and double, modify the lower bits directly
  // For long double, modify the first 8 bytes which contain the mantissa
  if constexpr (sizeof(T) == sizeof(float)) {
    uint32_t int_bits;
    std::memcpy(&int_bits, bits, sizeof(uint32_t));
    int_bits = (int_bits & ~uint32_t(0xFFFFF)) | (payload & 0xFFFFF);
    std::memcpy(bits, &int_bits, sizeof(uint32_t));
  } else {
    // For double and long double, modify the first 64 bits
    uint64_t int_bits;
    std::memcpy(&int_bits, bits, sizeof(uint64_t));
    int_bits = (int_bits & ~uint64_t(0xFFFFF)) | (payload & 0xFFFFF);
    std::memcpy(bits, &int_bits, sizeof(uint64_t));
  }

  std::memcpy(&nan, bits, sizeof(T));
  return nan;
}

// Test fetch_fmaximum for floating-point types
// LoadOp: () -> T - reads current value
// StoreOp: (T) -> void - stores a value
// FMaximumOp: (T, memory_order) -> T - performs fetch_fmaximum operation
template <class T, class LoadOp, class StoreOp, class FMaximumOp>
void test_fetch_fmaximum(LoadOp load, StoreOp store, FMaximumOp fmaximum) {
  static_assert(std::is_floating_point_v<T>);

  constexpr T nan = std::numeric_limits<T>::quiet_NaN();
  constexpr T inf = std::numeric_limits<T>::infinity();

  // Test basic fetch_fmaximum (update to larger value)
  {
    store(T(10.0));
    std::same_as<T> decltype(auto) old = fmaximum(T(20.0), std::memory_order_seq_cst);
    assert(old == T(10.0));
    assert(load() == T(20.0));
  }

  // Test with smaller value (no change)
  {
    store(T(10.0));
    std::same_as<T> decltype(auto) old = fmaximum(T(5.0), std::memory_order_seq_cst);
    assert(old == T(10.0));
    assert(load() == T(10.0));
  }

  // Test with negative values
  {
    store(T(-10.0));
    std::same_as<T> decltype(auto) old = fmaximum(T(-5.0), std::memory_order_seq_cst);
    assert(old == T(-10.0));
    assert(load() == T(-5.0));
  }

  {
    store(T(-5.0));
    std::same_as<T> decltype(auto) old = fmaximum(T(-10.0), std::memory_order_seq_cst);
    assert(old == T(-5.0));
    assert(load() == T(-5.0));
  }

  // Test NaN handling: propagate NaN
  {
    store(nan);
    std::same_as<T> decltype(auto) old = fmaximum(T(5.0), std::memory_order_seq_cst);
    assert(std::isnan(old));
    assert(std::isnan(load()));
  }

  {
    store(T(5.0));
    std::same_as<T> decltype(auto) old = fmaximum(nan, std::memory_order_seq_cst);
    assert(old == T(5.0));
    assert(std::isnan(load()));
  }

  // Both NaN: return NaN
  {
    store(nan);
    std::same_as<T> decltype(auto) old = fmaximum(nan, std::memory_order_seq_cst);
    assert(std::isnan(old));
    assert(std::isnan(load()));
  }

  // Test signed zero handling: -0.0 < +0.0, so max returns +0.0
  {
    store(T(-0.0));
    std::same_as<T> decltype(auto) old = fmaximum(T(+0.0), std::memory_order_seq_cst);
    assert(old == T(-0.0));
    assert(std::signbit(old));
    assert(!std::signbit(load()));
  }

  {
    store(T(+0.0));
    std::same_as<T> decltype(auto) old = fmaximum(T(-0.0), std::memory_order_seq_cst);
    assert(old == T(+0.0));
    assert(!std::signbit(old));
    assert(!std::signbit(load()));
  }

  // Test infinity
  {
    store(T(1.0));
    std::same_as<T> decltype(auto) old = fmaximum(inf, std::memory_order_seq_cst);
    assert(old == T(1.0));
    assert(load() == inf);
  }

  {
    store(-inf);
    std::same_as<T> decltype(auto) old = fmaximum(T(1.0), std::memory_order_seq_cst);
    assert(old == -inf);
    assert(load() == T(1.0));
  }

  // Test different memory orderings
  {
    store(T(8.0));
    std::same_as<T> decltype(auto) old = fmaximum(T(15.0), std::memory_order_relaxed);
    assert(old == T(8.0));
    assert(load() == T(15.0));
  }

  {
    store(T(3.0));
    std::same_as<T> decltype(auto) old = fmaximum(T(10.0), std::memory_order_acquire);
    assert(old == T(3.0));
    assert(load() == T(10.0));
  }

  {
    store(T(2.0));
    std::same_as<T> decltype(auto) old = fmaximum(T(10.0), std::memory_order_release);
    assert(old == T(2.0));
    assert(load() == T(10.0));
  }

  {
    store(T(7.0));
    std::same_as<T> decltype(auto) old = fmaximum(T(10.0), std::memory_order_acq_rel);
    assert(old == T(7.0));
    assert(load() == T(10.0));
  }

  // Test with different NaN representations
  {
    T nan1 = make_nan_with_payload<T>(1);

    // Store NaN with payload 1, fetch with non-NaN (propagates NaN)
    store(nan1);
    std::same_as<T> decltype(auto) old = fmaximum(T(5.0), std::memory_order_seq_cst);
    assert(std::isnan(old));
    assert(std::isnan(load()));
  }

  {
    T nan2 = make_nan_with_payload<T>(2);

    // Store non-NaN, fetch with NaN with payload 2 (propagates NaN)
    store(T(5.0));
    std::same_as<T> decltype(auto) old = fmaximum(nan2, std::memory_order_seq_cst);
    assert(old == T(5.0));
    assert(std::isnan(load()));
  }

  {
    T nan1 = make_nan_with_payload<T>(1);
    T nan2 = make_nan_with_payload<T>(2);

    // Store NaN with payload 1, fetch with NaN with payload 2 (propagates NaN)
    store(nan1);
    std::same_as<T> decltype(auto) old = fmaximum(nan2, std::memory_order_seq_cst);
    assert(std::isnan(old));
    assert(std::isnan(load()));
    // Result should be one of the NaN values (implementation-defined which)
  }
}

#endif // TEST_STD_ATOMICS_ATOMIC_FETCH_FMAXIMUM_HELPER_H
