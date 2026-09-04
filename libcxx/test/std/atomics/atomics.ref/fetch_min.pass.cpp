//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// XFAIL: !has-64-bit-atomics

// integral-type fetch_min(integral-type, memory_order = memory_order::seq_cst) const noexcept;
// T* fetch_min(T*, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "atomic_fetch_min_helper.h"
#include "atomic_helpers.h"
#include "test_macros.h"

template <typename T>
concept has_fetch_min = requires(std::atomic_ref<T> const& a, T v) {
  { a.fetch_min(v) } -> std::same_as<T>;
  { a.fetch_min(v, std::memory_order::relaxed) } -> std::same_as<T>;
};

template <typename T>
struct TestDoesNotHaveFetchMin {
  void operator()() const { static_assert(!has_fetch_min<T>); }
};

template <typename T>
struct TestFetchMin {
  void operator()() const {
    static_assert((std::is_integral_v<T> || std::is_pointer_v<T>) && has_fetch_min<T>);

    if constexpr (std::is_integral_v<T>) {
      alignas(std::atomic_ref<T>::required_alignment) T x{};
      std::atomic_ref<T> const a(x);

      auto load  = [&]() { return x; };
      auto store = [&](T val) { x = val; };
      auto min   = [&](T val, auto order) { return a.fetch_min(val, order); };

      ASSERT_NOEXCEPT(a.fetch_min(T(0), std::memory_order_seq_cst));
      test_fetch_min_integral<T>(load, store, min);

    } else if constexpr (std::is_pointer_v<T>) {
      using U = std::remove_pointer_t<T>;
      U arr[5]{};
      alignas(std::atomic_ref<T>::required_alignment) T p{};
      std::atomic_ref<T> const a(p);

      auto load  = [&]() { return p; };
      auto store = [&](T val) { p = val; };
      auto min   = [&](T val, auto order) { return a.fetch_min(val, order); };

      ASSERT_NOEXCEPT(a.fetch_min(&arr[0], std::memory_order_seq_cst));
      test_fetch_min_pointer<U>(&arr[0], &arr[2], &arr[4], load, store, min);
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestFetchMin>()();
  TestEachPointerType<TestFetchMin>()();

  TestEachFloatingPointType<TestDoesNotHaveFetchMin>()();

  TestDoesNotHaveFetchMin<bool>()();
  TestDoesNotHaveFetchMin<UserAtomicType>()();
  TestDoesNotHaveFetchMin<LargeUserAtomicType>()();

  return 0;
}
