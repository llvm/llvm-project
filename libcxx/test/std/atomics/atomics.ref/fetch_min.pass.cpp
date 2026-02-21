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

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

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
      alignas(std::atomic_ref<T>::required_alignment) T x(T(3));
      std::atomic_ref<T> const a(x);

      {
        std::same_as<T> decltype(auto) y = a.fetch_min(T(2));
        assert(y == T(3));
        assert(x == T(2));
        y = a.fetch_min(T(4));
        assert(y == T(2));
        assert(x == T(2));
        ASSERT_NOEXCEPT(a.fetch_min(T(0)));
      }

      {
        std::same_as<T> decltype(auto) y = a.fetch_min(T(1), std::memory_order_relaxed);
        assert(y == T(2));
        assert(x == T(1));
        y = a.fetch_min(T(4));
        assert(y == T(1));
        assert(x == T(1));
        ASSERT_NOEXCEPT(a.fetch_min(T(0), std::memory_order_relaxed));
      }
    } else if constexpr (std::is_pointer_v<T>) {
      using U = std::remove_pointer_t<T>;
      U t[9]  = {};
      alignas(std::atomic_ref<T>::required_alignment) T x{&t[3]};
      std::atomic_ref<T> const a(x);

      {
        std::same_as<T> decltype(auto) y = a.fetch_min(&t[2]);
        assert(y == &t[3]);
        assert(x == &t[2]);
        y = a.fetch_min(&t[4]);
        assert(y == &t[2]);
        assert(x == &t[2]);
        ASSERT_NOEXCEPT(a.fetch_min(&t[0]));
      }

      {
        std::same_as<T> decltype(auto) y = a.fetch_min(&t[1], std::memory_order_relaxed);
        assert(y == &t[2]);
        assert(a == &t[1]);
        y = a.fetch_min(&t[4]);
        assert(y == &t[1]);
        assert(x == &t[1]);
        ASSERT_NOEXCEPT(a.fetch_min(&t[0], std::memory_order_relaxed));
      }
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestFetchMin>()();

  TestEachFloatingPointType<TestDoesNotHaveFetchMin>()();

  TestEachPointerType<TestFetchMin>()();

  TestDoesNotHaveFetchMin<bool>()();
  TestDoesNotHaveFetchMin<UserAtomicType>()();
  TestDoesNotHaveFetchMin<LargeUserAtomicType>()();

  return 0;
}
