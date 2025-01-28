//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// integral-type fetch_add(integral-type, memory_order = memory_order::seq_cst) const noexcept;
// floating-point-type fetch_add(floating-point-type, memory_order = memory_order::seq_cst) const noexcept;
// T* fetch_add(difference_type, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "atomic_helpers.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
concept has_fetch_add = requires {
  std::declval<T const>().fetch_add(std::declval<T>());
  std::declval<T const>().fetch_add(std::declval<T>(), std::declval<std::memory_order>());
};

template <typename T>
struct TestDoesNotHaveFetchAdd {
  void operator()() const { static_assert(!has_fetch_add<std::atomic_ref<T>>); }
};

template <typename T>
struct TestFetchAdd {
  void operator()() const {
    if constexpr (std::is_arithmetic_v<T>) {
      T x(T(1));
      std::atomic_ref<T> const a(x);

      {
        std::same_as<T> decltype(auto) y = a.fetch_add(T(2));
        assert(y == T(1));
        assert(x == T(3));
        ASSERT_NOEXCEPT(a.fetch_add(T(0)));
      }

      {
        std::same_as<T> decltype(auto) y = a.fetch_add(T(4), std::memory_order_relaxed);
        assert(y == T(3));
        assert(x == T(7));
        ASSERT_NOEXCEPT(a.fetch_add(T(0), std::memory_order_relaxed));
      }
    } else if constexpr (std::is_pointer_v<T>) {
      using U = std::remove_pointer_t<T>;
      U t[9]  = {};
      T p{&t[1]};
      std::atomic_ref<T> const a(p);

      {
        std::same_as<T> decltype(auto) y = a.fetch_add(2);
        assert(y == &t[1]);
        assert(a == &t[3]);
        ASSERT_NOEXCEPT(a.fetch_add(0));
      }

      {
        std::same_as<T> decltype(auto) y = a.fetch_add(4, std::memory_order_relaxed);
        assert(y == &t[3]);
        assert(a == &t[7]);
        ASSERT_NOEXCEPT(a.fetch_add(0, std::memory_order_relaxed));
      }
    } else {
      static_assert(std::is_void_v<T>);
    }

    // memory_order::release
    {
      auto fetch_add = [](std::atomic_ref<T> const& x, T old_val, T new_val) {
        x.fetch_add(new_val - old_val, std::memory_order::release);
      };
      auto load = [](std::atomic_ref<T> const& x) { return x.load(std::memory_order::acquire); };
      test_acquire_release<T>(fetch_add, load);
    }

    // memory_order::seq_cst
    {
      auto fetch_add_no_arg = [](std::atomic_ref<T> const& x, T old_val, T new_val) { x.fetch_add(new_val - old_val); };
      auto fetch_add_with_order = [](std::atomic_ref<T> const& x, T old_val, T new_val) {
        x.fetch_add(new_val - old_val, std::memory_order::seq_cst);
      };
      auto load = [](std::atomic_ref<T> const& x) { return x.load(); };
      test_seq_cst<T>(fetch_add_no_arg, load);
      test_seq_cst<T>(fetch_add_with_order, load);
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestFetchAdd>()();

  TestFetchAdd<float>()();
  TestFetchAdd<double>()();

  TestEachPointerType<TestFetchAdd>()();

  TestDoesNotHaveFetchAdd<bool>()();
  TestDoesNotHaveFetchAdd<UserAtomicType>()();
  TestDoesNotHaveFetchAdd<LargeUserAtomicType>()();

  return 0;
}
