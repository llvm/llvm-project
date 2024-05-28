//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics
// XFAIL: !has-1024-bit-atomics

// MSVC warning C4310: cast truncates constant value
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd4310

// bool compare_exchange_strong(T&, T, memory_order, memory_order) const noexcept;
// bool compare_exchange_strong(T&, T, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
struct TestCompareExchangeStrong {
  void operator()() const {
    {
      T x(T(1));
      std::atomic_ref<T> const a(x);

      T t(T(1));
      std::same_as<bool> decltype(auto) y = a.compare_exchange_strong(t, T(2));
      assert(y == true);
      assert(a == T(2));
      assert(t == T(1));
      y = a.compare_exchange_strong(t, T(3));
      assert(y == false);
      assert(a == T(2));
      assert(t == T(2));

      ASSERT_NOEXCEPT(a.compare_exchange_strong(t, T(2)));
    }
    {
      T x(T(1));
      std::atomic_ref<T> const a(x);

      T t(T(1));
      std::same_as<bool> decltype(auto) y = a.compare_exchange_strong(t, T(2), std::memory_order_seq_cst);
      assert(y == true);
      assert(a == T(2));
      assert(t == T(1));
      y = a.compare_exchange_strong(t, T(3), std::memory_order_seq_cst);
      assert(y == false);
      assert(a == T(2));
      assert(t == T(2));

      ASSERT_NOEXCEPT(a.compare_exchange_strong(t, T(2), std::memory_order_seq_cst));
    }
    {
      T x(T(1));
      std::atomic_ref<T> const a(x);

      T t(T(1));
      std::same_as<bool> decltype(auto) y =
          a.compare_exchange_strong(t, T(2), std::memory_order_release, std::memory_order_relaxed);
      assert(y == true);
      assert(a == T(2));
      assert(t == T(1));
      y = a.compare_exchange_strong(t, T(3), std::memory_order_release, std::memory_order_relaxed);
      assert(y == false);
      assert(a == T(2));
      assert(t == T(2));

      ASSERT_NOEXCEPT(a.compare_exchange_strong(t, T(2), std::memory_order_release, std::memory_order_relaxed));
    }

    // success memory_order::release
    {
      auto store = [](std::atomic_ref<T> const& x, T old_val, T new_val) {
        auto r = x.compare_exchange_strong(old_val, new_val, std::memory_order::release, std::memory_order::relaxed);
        assert(r);
      };

      auto load = [](std::atomic_ref<T> const& x) { return x.load(std::memory_order::acquire); };
      test_acquire_release<T>(store, load);

      auto store_one_arg = [](std::atomic_ref<T> const& x, T old_val, T new_val) {
        auto r = x.compare_exchange_strong(old_val, new_val, std::memory_order::release);
        assert(r);
      };
      test_acquire_release<T>(store_one_arg, load);
    }

    // success memory_order::acquire
    {
      auto store = [](std::atomic_ref<T> const& x, T, T new_val) { x.store(new_val, std::memory_order::release); };

      auto load = [](std::atomic_ref<T> const& x) {
        auto val = x.load(std::memory_order::relaxed);
        while (!x.compare_exchange_strong(val, val, std::memory_order::acquire, std::memory_order::relaxed)) {
        }
        return val;
      };
      test_acquire_release<T>(store, load);

      auto load_one_arg = [](std::atomic_ref<T> const& x) {
        auto val = x.load(std::memory_order::relaxed);
        while (!x.compare_exchange_strong(val, val, std::memory_order::acquire)) {
        }
        return val;
      };
      test_acquire_release<T>(store, load_one_arg);
    }

    // success memory_order::acq_rel
    {
      auto store = [](std::atomic_ref<T> const& x, T old_val, T new_val) {
        auto r = x.compare_exchange_strong(old_val, new_val, std::memory_order::acq_rel, std::memory_order::relaxed);
        assert(r);
      };
      auto load = [](std::atomic_ref<T> const& x) {
        auto val = x.load(std::memory_order::relaxed);
        while (!x.compare_exchange_strong(val, val, std::memory_order::acq_rel, std::memory_order::relaxed)) {
        }
        return val;
      };
      test_acquire_release<T>(store, load);

      auto store_one_arg = [](std::atomic_ref<T> const& x, T old_val, T new_val) {
        auto r = x.compare_exchange_strong(old_val, new_val, std::memory_order::acq_rel);
        assert(r);
      };
      auto load_one_arg = [](std::atomic_ref<T> const& x) {
        auto val = x.load(std::memory_order::relaxed);
        while (!x.compare_exchange_strong(val, val, std::memory_order::acq_rel)) {
        }
        return val;
      };
      test_acquire_release<T>(store_one_arg, load_one_arg);
    }

    // success memory_order::seq_cst
    {
      auto store = [](std::atomic_ref<T> const& x, T old_val, T new_val) {
        auto r = x.compare_exchange_strong(old_val, new_val, std::memory_order::seq_cst, std::memory_order::relaxed);
        assert(r);
      };
      auto load = [](std::atomic_ref<T> const& x) {
        auto val = x.load(std::memory_order::relaxed);
        while (!x.compare_exchange_strong(val, val, std::memory_order::seq_cst, std::memory_order::relaxed)) {
        }
        return val;
      };
      test_seq_cst<T>(store, load);

      auto store_one_arg = [](std::atomic_ref<T> const& x, T old_val, T new_val) {
        auto r = x.compare_exchange_strong(old_val, new_val, std::memory_order::seq_cst);
        assert(r);
      };
      auto load_one_arg = [](std::atomic_ref<T> const& x) {
        auto val = x.load(std::memory_order::relaxed);
        while (!x.compare_exchange_strong(val, val, std::memory_order::seq_cst)) {
        }
        return val;
      };
      test_seq_cst<T>(store_one_arg, load_one_arg);
    }

    // failure memory_order::acquire
    {
      auto store = [](std::atomic_ref<T> const& x, T, T new_val) { x.store(new_val, std::memory_order::release); };
      auto load  = [](std::atomic_ref<T> const& x) {
        auto result = x.load(std::memory_order::relaxed);
        T unexpected(T(255));
        bool r =
            x.compare_exchange_strong(unexpected, unexpected, std::memory_order::relaxed, std::memory_order::acquire);
        assert(!r);
        return result;
      };
      test_acquire_release<T>(store, load);

      auto load_one_arg = [](std::atomic_ref<T> const& x) {
        auto result = x.load(std::memory_order::relaxed);
        T unexpected(T(255));
        bool r = x.compare_exchange_strong(unexpected, unexpected, std::memory_order::acquire);
        assert(!r);
        return result;
      };
      test_acquire_release<T>(store, load_one_arg);

      // acq_rel replaced by acquire
      auto load_one_arg_acq_rel = [](std::atomic_ref<T> const& x) {
        auto result = x.load(std::memory_order::relaxed);
        T unexpected(T(255));
        bool r = x.compare_exchange_strong(unexpected, unexpected, std::memory_order::acq_rel);
        assert(!r);
        return result;
      };
      test_acquire_release<T>(store, load_one_arg_acq_rel);
    }

    // failure memory_order::seq_cst
    {
      auto store = [](std::atomic_ref<T> const& x, T, T new_val) { x.store(new_val, std::memory_order::seq_cst); };
      auto load  = [](std::atomic_ref<T> const& x) {
        auto result = x.load(std::memory_order::relaxed);
        T unexpected(T(255));
        bool r =
            x.compare_exchange_strong(unexpected, unexpected, std::memory_order::relaxed, std::memory_order::seq_cst);
        assert(!r);
        return result;
      };
      test_seq_cst<T>(store, load);
    }
  }
};

int main(int, char**) {
  TestEachAtomicType<TestCompareExchangeStrong>()();
  return 0;
}
