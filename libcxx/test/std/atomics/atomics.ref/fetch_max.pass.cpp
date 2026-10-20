//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// XFAIL: !has-64-bit-atomics

// integral-type fetch_max(integral-type, memory_order = memory_order::seq_cst) const noexcept;
// T* fetch_max(T*, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "atomic_fetch_max_helper.h"
#include "atomic_helpers.h"
#include "test_macros.h"

template <typename T>
concept has_fetch_max = requires(std::atomic_ref<T> const& a, T v) {
  { a.fetch_max(v) } -> std::same_as<T>;
  { a.fetch_max(v, std::memory_order::relaxed) } -> std::same_as<T>;
};

template <typename T>
struct TestDoesNotHaveFetchMax {
  void operator()() const { static_assert(!has_fetch_max<T>); }
};

template <typename T>
struct TestFetchMax {
  void operator()() const {
    static_assert((std::is_integral_v<T> || std::is_pointer_v<T>) && has_fetch_max<T>);

    if constexpr (std::is_integral_v<T>) {
      alignas(std::atomic_ref<T>::required_alignment) T x{};
      std::atomic_ref<T> const a(x);

      auto load  = [&]() { return x; };
      auto store = [&](T val) { x = val; };
      auto max   = [&](T val, auto order) { return a.fetch_max(val, order); };

      ASSERT_NOEXCEPT(a.fetch_max(T(0), std::memory_order_seq_cst));
      test_fetch_max_integral<T>(load, store, max);

    } else if constexpr (std::is_pointer_v<T>) {
      using U = std::remove_pointer_t<T>;
      U arr[5]{};
      alignas(std::atomic_ref<T>::required_alignment) T p{};
      std::atomic_ref<T> const a(p);

      auto load  = [&]() { return p; };
      auto store = [&](T val) { p = val; };
      auto max   = [&](T val, auto order) { return a.fetch_max(val, order); };

      ASSERT_NOEXCEPT(a.fetch_max(&arr[0], std::memory_order_seq_cst));
      test_fetch_max_pointer<U>(&arr[0], &arr[2], &arr[4], load, store, max);
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestFetchMax>()();
  TestEachPointerType<TestFetchMax>()();

  TestEachFloatingPointType<TestDoesNotHaveFetchMax>()();

  TestDoesNotHaveFetchMax<bool>()();
  TestDoesNotHaveFetchMax<UserAtomicType>()();
  TestDoesNotHaveFetchMax<LargeUserAtomicType>()();

  return 0;
}
