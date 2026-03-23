//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// XFAIL: !has-64-bit-atomics

// integral-type fetch_min(integral-type, memory_order = memory_order::seq_cst) const noexcept;
// T* fetch_min(T*, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_macros.h"
#include "atomic_helpers.h"
#include "../atomics.types.operations/atomics.types.operations.req/atomic_fetch_min_helper.h"

template <typename T>
concept has_fetch_min = requires(const T& t, typename T::value_type v, std::memory_order m) {
  t.fetch_min(v);
  t.fetch_min(v, m);
};

template <typename T>
struct TestDoesNotHaveFetchMin {
  void operator()() const { static_assert(!has_fetch_min<std::atomic_ref<T>>); }
};

template <typename T>
struct TestFetchMin {
  void operator()() const {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      alignas(std::atomic_ref<T>::required_alignment) T x;
      std::atomic_ref<T> const a(x);

      auto load  = [&]() { return x; };
      auto store = [&](T val) { x = val; };
      auto min   = [&](T val, auto order) { return a.fetch_min(val, order); };

      ASSERT_NOEXCEPT(a.fetch_min(T(0), std::memory_order_seq_cst));
      test_fetch_min_integral<T>(load, store, min);

    } else if constexpr (std::is_pointer_v<T>) {
      using U = std::remove_pointer_t<T>;
      U t[9]  = {};
      alignas(std::atomic_ref<T>::required_alignment) T p;
      std::atomic_ref<T> const a(p);

      auto load  = [&]() { return p; };
      auto store = [&](T val) { p = val; };
      auto min   = [&](T val, auto order) { return a.fetch_min(val, order); };

      ASSERT_NOEXCEPT(a.fetch_min(&t[0], std::memory_order_seq_cst));
      test_fetch_min_pointer<U>(&t[0], &t[2], &t[4], load, store, min);

    } else {
      static_assert(std::is_void_v<T>);
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestFetchMin>()();
  TestEachPointerType<TestFetchMin>()();

  TestDoesNotHaveFetchMin<bool>{}();
  TestDoesNotHaveFetchMin<float>{}();
  TestDoesNotHaveFetchMin<double>{}();

  return 0;
}
