//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics

// bool compare_exchange_weak(T& expected, T desired,
//                            memory_order success, memory_order failure) volatile noexcept;
// bool compare_exchange_weak(T& expected, T desired,
//                            memory_order success, memory_order failure) noexcept;
// bool compare_exchange_weak(T& expected, T desired,
//                            memory_order order = memory_order::seq_cst) volatile noexcept;
// bool compare_exchange_weak(T& expected, T desired,
//                            memory_order order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "test_helper.h"
#include "test_macros.h"

template <class T, class... Args>
concept HasVolatileCompareExchangeWeak =
    requires(volatile std::atomic<T>& a, T t, Args... args) { a.compare_exchange_weak(t, t, args...); };

template <class T, template <class> class MaybeVolatile, class... Args>
concept HasNoexceptCompareExchangeWeak = requires(MaybeVolatile<std::atomic<T>>& a, T t, Args... args) {
  { a.compare_exchange_weak(t, t, args...) } noexcept;
};

template <class T, template <class> class MaybeVolatile = std::type_identity_t, class... MemoryOrder>
void testBasic(MemoryOrder... memory_order) {
  // Uncomment the test after P1831R1 is implemented
  // static_assert(HasVolatileCompareExchangeWeak<T, MemoryOrder...> == std::atomic<T>::is_always_lock_free);
  static_assert(HasNoexceptCompareExchangeWeak<T, MaybeVolatile, MemoryOrder...>);

  // compare pass
  {
    MaybeVolatile<std::atomic<T>> a(T(1.2));
    T expected(T(1.2));
    const T desired(T(2.3));
    std::same_as<bool> decltype(auto) r = a.compare_exchange_weak(expected, desired, memory_order...);

    // could be false spuriously
    if (r) {
      assert(a.load() == desired);
    }
    // if r is true, expected should be unmodified (1.2)
    // if r is false, the original value of a (1.2) is written to expected
    assert(expected == T(1.2));
  }

  // compare fail
  {
    MaybeVolatile<std::atomic<T>> a(T(1.2));
    T expected(1.5);
    const T desired(T(2.3));
    std::same_as<bool> decltype(auto) r = a.compare_exchange_weak(expected, desired, memory_order...);

    assert(!r);
    assert(a.load() == T(1.2));

    // bug
    // TODO https://github.com/llvm/llvm-project/issues/47978
    if constexpr (!std::same_as<T, long double>) {
      assert(expected == T(1.2));
    }
  }
}

template <class T, template <class> class MaybeVolatile = std::type_identity_t>
void test_impl() {
  testBasic<T, MaybeVolatile>();
  testBasic<T, MaybeVolatile>(std::memory_order::relaxed);
  testBasic<T, MaybeVolatile>(std::memory_order::relaxed, std::memory_order_relaxed);

  // test success memory order release
  {
    auto store = [](MaybeVolatile<std::atomic<T>>& x, T old_val, T new_val) {
      // could fail spuriously, so put it in a loop
      while (!x.compare_exchange_weak(old_val, new_val, std::memory_order::release, std::memory_order_relaxed)) {
      }
    };

    auto load = [](MaybeVolatile<std::atomic<T>>& x) { return x.load(std::memory_order::acquire); };
    test_acquire_release<T, MaybeVolatile>(store, load);

    auto store_one_arg = [](MaybeVolatile<std::atomic<T>>& x, T old_val, T new_val) {
      // could fail spuriously, so put it in a loop
      while (!x.compare_exchange_weak(old_val, new_val, std::memory_order::release)) {
      }
    };
    test_acquire_release<T, MaybeVolatile>(store_one_arg, load);
  }

  // test success memory order acquire
  {
    auto store = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x.store(new_val, std::memory_order::release); };
    auto load  = [](MaybeVolatile<std::atomic<T>>& x) {
      auto val = x.load(std::memory_order::relaxed);
      while (!x.compare_exchange_weak(val, val, std::memory_order::acquire, std::memory_order_relaxed)) {
      }
      return val;
    };
    test_acquire_release<T, MaybeVolatile>(store, load);

    auto load_one_arg = [](MaybeVolatile<std::atomic<T>>& x) {
      auto val = x.load(std::memory_order::relaxed);
      while (!x.compare_exchange_weak(val, val, std::memory_order::acquire)) {
      }
      return val;
    };
    test_acquire_release<T, MaybeVolatile>(store, load_one_arg);
  }

  // test success memory order acq_rel
  {
    auto store = [](MaybeVolatile<std::atomic<T>>& x, T old_val, T new_val) {
      // could fail spuriously, so put it in a loop
      while (!x.compare_exchange_weak(old_val, new_val, std::memory_order::acq_rel, std::memory_order_relaxed)) {
      }
    };
    auto load = [](MaybeVolatile<std::atomic<T>>& x) {
      auto val = x.load(std::memory_order::relaxed);
      while (!x.compare_exchange_weak(val, val, std::memory_order::acq_rel, std::memory_order_relaxed)) {
      }
      return val;
    };
    test_acquire_release<T, MaybeVolatile>(store, load);

    auto store_one_arg = [](MaybeVolatile<std::atomic<T>>& x, T old_val, T new_val) {
      // could fail spuriously, so put it in a loop
      while (!x.compare_exchange_weak(old_val, new_val, std::memory_order::acq_rel)) {
      }
    };
    auto load_one_arg = [](MaybeVolatile<std::atomic<T>>& x) {
      auto val = x.load(std::memory_order::relaxed);
      while (!x.compare_exchange_weak(val, val, std::memory_order::acq_rel)) {
      }
      return val;
    };
    test_acquire_release<T, MaybeVolatile>(store_one_arg, load_one_arg);
  }

  // test success memory seq_cst
  {
    auto store = [](MaybeVolatile<std::atomic<T>>& x, T old_val, T new_val) {
      // could fail spuriously, so put it in a loop
      while (!x.compare_exchange_weak(old_val, new_val, std::memory_order::seq_cst, std::memory_order_relaxed)) {
      }
    };
    auto load = [](MaybeVolatile<std::atomic<T>>& x) {
      auto val = x.load(std::memory_order::relaxed);
      while (!x.compare_exchange_weak(val, val, std::memory_order::seq_cst, std::memory_order_relaxed)) {
      }
      return val;
    };
    test_seq_cst<T, MaybeVolatile>(store, load);

    auto store_one_arg = [](MaybeVolatile<std::atomic<T>>& x, T old_val, T new_val) {
      // could fail spuriously, so put it in a loop
      while (!x.compare_exchange_weak(old_val, new_val, std::memory_order::seq_cst, std::memory_order_relaxed)) {
      }
    };
    auto load_one_arg = [](MaybeVolatile<std::atomic<T>>& x) {
      auto val = x.load(std::memory_order::relaxed);
      while (!x.compare_exchange_weak(val, val, std::memory_order::seq_cst, std::memory_order_relaxed)) {
      }
      return val;
    };
    test_seq_cst<T, MaybeVolatile>(store_one_arg, load_one_arg);
  }

  // test fail memory order acquire
  {
    auto store = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x.store(new_val, std::memory_order::release); };
    auto load  = [](MaybeVolatile<std::atomic<T>>& x) {
      auto result = x.load(std::memory_order::relaxed);
      T unexpected(T(-9999.99));
      bool r = x.compare_exchange_weak(unexpected, unexpected, std::memory_order_relaxed, std::memory_order_acquire);
      assert(!r);
      return result;
    };
    test_acquire_release<T, MaybeVolatile>(store, load);

    auto load_one_arg = [](MaybeVolatile<std::atomic<T>>& x) {
      auto result = x.load(std::memory_order::relaxed);
      T unexpected(T(-9999.99));
      bool r = x.compare_exchange_weak(unexpected, unexpected, std::memory_order_acquire);
      assert(!r);
      return result;
    };
    test_acquire_release<T, MaybeVolatile>(store, load_one_arg);

    // acq_rel replaced by acquire
    auto load_one_arg_acq_rel = [](MaybeVolatile<std::atomic<T>>& x) {
      auto result = x.load(std::memory_order::relaxed);
      T unexpected(T(-9999.99));
      bool r = x.compare_exchange_weak(unexpected, unexpected, std::memory_order_acq_rel);
      assert(!r);
      return result;
    };
    test_acquire_release<T, MaybeVolatile>(store, load_one_arg_acq_rel);
  }

  // test fail memory order seq_cst
  {
    auto store = [](MaybeVolatile<std::atomic<T>>& x, T, T new_val) { x.store(new_val, std::memory_order::seq_cst); };
    auto load  = [](MaybeVolatile<std::atomic<T>>& x) {
      auto result = x.load(std::memory_order::relaxed);
      T unexpected(T(-9999.99));
      bool r = x.compare_exchange_weak(unexpected, unexpected, std::memory_order_relaxed, std::memory_order::seq_cst);
      assert(!r);
      return result;
    };
    test_seq_cst<T, MaybeVolatile>(store, load);
  }
}

template <class T>
void test() {
  test_impl<T>();
  if constexpr (std::atomic<T>::is_always_lock_free) {
    test_impl<T, std::add_volatile_t>();
  }
}

int main(int, char**) {
  test<float>();
  test<double>();

  // TODO https://github.com/llvm/llvm-project/issues/47978
  // test<long double>();

  return 0;
}
