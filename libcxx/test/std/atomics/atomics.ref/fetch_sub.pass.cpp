//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// integral-type fetch_sub(integral-type, memory_order = memory_order::seq_cst) const noexcept;
// floating-point-type fetch_sub(floating-point-type, memory_order = memory_order::seq_cst) const noexcept;
// T* fetch_sub(difference_type, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
concept has_fetch_sub = requires {
  std::declval<T const>().fetch_sub(std::declval<T>());
  std::declval<T const>().fetch_sub(std::declval<T>(), std::declval<std::memory_order>());
};

template <typename T>
struct TestDoesNotHaveFetchSub {
  void operator()() const { static_assert(!has_fetch_sub<std::atomic_ref<T>>); }
};

template <typename T>
struct TestFetchSub {
  void operator()() const {
    if constexpr (std::is_arithmetic_v<T>) {
      T x(T(7));
      std::atomic_ref<T> const a(x);

      {
        std::same_as<T> decltype(auto) y = a.fetch_sub(T(4));
        assert(y == T(7));
        assert(x == T(3));
        ASSERT_NOEXCEPT(a.fetch_sub(T(0)));
      }

      {
        std::same_as<T> decltype(auto) y = a.fetch_sub(T(2), std::memory_order_relaxed);
        assert(y == T(3));
        assert(x == T(1));
        ASSERT_NOEXCEPT(a.fetch_sub(T(0), std::memory_order_relaxed));
      }
    } else if constexpr (std::is_pointer_v<T>) {
      using U = std::remove_pointer_t<T>;
      U t[9]  = {};
      T p{&t[7]};
      std::atomic_ref<T> const a(p);

      {
        std::same_as<T> decltype(auto) y = a.fetch_sub(4);
        assert(y == &t[7]);
        assert(a == &t[3]);
        ASSERT_NOEXCEPT(a.fetch_sub(0));
      }

      {
        std::same_as<T> decltype(auto) y = a.fetch_sub(2, std::memory_order_relaxed);
        assert(y == &t[3]);
        assert(a == &t[1]);
        ASSERT_NOEXCEPT(a.fetch_sub(0, std::memory_order_relaxed));
      }
    } else {
      static_assert(std::is_void_v<T>);
    }

    // memory_order::release
    {
      auto fetch_sub = [](std::atomic_ref<T> const& x, T old_val, T new_val) {
        x.fetch_sub(old_val - new_val, std::memory_order::release);
      };
      auto load = [](std::atomic_ref<T> const& x) { return x.load(std::memory_order::acquire); };
      test_acquire_release<T>(fetch_sub, load);
    }

    // memory_order::seq_cst
    {
      auto fetch_sub_no_arg = [](std::atomic_ref<T> const& x, T old_val, T new_val) { x.fetch_sub(old_val - new_val); };
      auto fetch_sub_with_order = [](std::atomic_ref<T> const& x, T old_val, T new_val) {
        x.fetch_sub(old_val - new_val, std::memory_order::seq_cst);
      };
      auto load = [](std::atomic_ref<T> const& x) { return x.load(); };
      test_seq_cst<T>(fetch_sub_no_arg, load);
      test_seq_cst<T>(fetch_sub_with_order, load);
    }
  }
};

int main(int, char**) {
  TestEachIntegralType<TestFetchSub>()();

  TestFetchSub<float>()();
  TestFetchSub<double>()();

  TestEachPointerType<TestFetchSub>()();

  TestDoesNotHaveFetchSub<bool>()();
  TestDoesNotHaveFetchSub<UserAtomicType>()();
  TestDoesNotHaveFetchSub<LargeUserAtomicType>()();

  return 0;
}
