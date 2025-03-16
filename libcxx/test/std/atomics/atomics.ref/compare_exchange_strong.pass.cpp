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

template <typename T, typename U>
concept has_compare_exchange_strong_1 =
    requires(T const& x, U& expected, U desired) { x.compare_exchange_strong(expected, desired); };
template <typename T, typename U>
concept has_compare_exchange_strong_2 = requires(T const& x, U& expected, U desired, std::memory_order order) {
  x.compare_exchange_strong(expected, desired, order);
};
template <typename T, typename U>
concept has_compare_exchange_strong_3 =
    requires(T const& x, U& expected, U desired, std::memory_order success, std::memory_order failure) {
      x.compare_exchange_strong(expected, desired, success, failure);
    };

template <typename T, typename U>
concept has_compare_exchange_strong =
    has_compare_exchange_strong_1<T, U> && has_compare_exchange_strong_2<T, U> && has_compare_exchange_strong_3<T, U>;

template <typename T, typename U>
concept does_not_have_compare_exchange_strong =
    !has_compare_exchange_strong_1<T, U> && !has_compare_exchange_strong_2<T, U> &&
    !has_compare_exchange_strong_3<T, U>;

template <typename T>
struct TestCompareExchangeStrong {
  void operator()() const {
    using Unqualified = std::remove_cv_t<T>;
    static_assert(has_compare_exchange_strong<std::atomic_ref<T>, Unqualified> == !std::is_const_v<T>);
    static_assert(does_not_have_compare_exchange_strong<std::atomic_ref<T>, Unqualified> == std::is_const_v<T>);

    if constexpr (!std::is_const_v<T>) {
      {
        T x(Unqualified(1));
        std::atomic_ref<T> const a(x);

        Unqualified t(Unqualified(1));
        std::same_as<bool> decltype(auto) y = a.compare_exchange_strong(t, Unqualified(2));
        assert(y == true);
        assert(a == Unqualified(2));
        assert(t == Unqualified(1));
        y = a.compare_exchange_strong(t, Unqualified(3));
        assert(y == false);
        assert(a == Unqualified(2));
        assert(t == Unqualified(2));

        ASSERT_NOEXCEPT(a.compare_exchange_strong(t, T(2)));
      }
      {
        T x(Unqualified(1));
        std::atomic_ref<T> const a(x);

        Unqualified t(Unqualified(1));
        std::same_as<bool> decltype(auto) y = a.compare_exchange_strong(t, Unqualified(2), std::memory_order_seq_cst);
        assert(y == true);
        assert(a == Unqualified(2));
        assert(t == Unqualified(1));
        y = a.compare_exchange_strong(t, T(3), std::memory_order_seq_cst);
        assert(y == false);
        assert(a == Unqualified(2));
        assert(t == Unqualified(2));

        ASSERT_NOEXCEPT(a.compare_exchange_strong(t, Unqualified(2), std::memory_order_seq_cst));
      }
      {
        T x(Unqualified(1));
        std::atomic_ref<T> const a(x);

        Unqualified t(Unqualified(1));
        std::same_as<bool> decltype(auto) y =
            a.compare_exchange_strong(t, Unqualified(2), std::memory_order_release, std::memory_order_relaxed);
        assert(y == true);
        assert(a == Unqualified(2));
        assert(t == Unqualified(1));
        y = a.compare_exchange_strong(t, Unqualified(3), std::memory_order_release, std::memory_order_relaxed);
        assert(y == false);
        assert(a == Unqualified(2));
        assert(t == Unqualified(2));

        ASSERT_NOEXCEPT(
            a.compare_exchange_strong(t, Unqualified(2), std::memory_order_release, std::memory_order_relaxed));
      }

      // success memory_order::release
      {
        auto store = [](std::atomic_ref<T> const& x, T const& old_val, T const& new_val) {
          auto r = x.compare_exchange_strong(
              const_cast<Unqualified&>(old_val),
              const_cast<Unqualified const&>(new_val),
              std::memory_order::release,
              std::memory_order::relaxed);
          assert(r);
        };

        auto load = [](std::atomic_ref<T> const& x) { return x.load(std::memory_order::acquire); };
        test_acquire_release<T>(store, load);
        auto store_one_arg = [](std::atomic_ref<T> const& x, T const& old_val, T const& new_val) {
          auto r = x.compare_exchange_strong(
              const_cast<Unqualified&>(old_val), const_cast<Unqualified const&>(new_val), std::memory_order::release);
          assert(r);
        };
        test_acquire_release<T>(store_one_arg, load);
      }

      // success memory_order::acquire
      {
        auto store = [](std::atomic_ref<T> const& x, T const&, T const& new_val) {
          x.store(const_cast<Unqualified const&>(new_val), std::memory_order::release);
        };

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
        auto store = [](std::atomic_ref<T> const& x, T const& old_val, T const& new_val) {
          auto r = x.compare_exchange_strong(
              const_cast<Unqualified&>(old_val),
              const_cast<Unqualified const&>(new_val),
              std::memory_order::acq_rel,
              std::memory_order::relaxed);
          assert(r);
        };
        auto load = [](std::atomic_ref<T> const& x) {
          auto val = x.load(std::memory_order::relaxed);
          while (!x.compare_exchange_strong(val, val, std::memory_order::acq_rel, std::memory_order::relaxed)) {
          }
          return val;
        };
        test_acquire_release<T>(store, load);

        auto store_one_arg = [](std::atomic_ref<T> const& x, T const& old_val, T const& new_val) {
          auto r = x.compare_exchange_strong(
              const_cast<Unqualified&>(old_val), const_cast<Unqualified const&>(new_val), std::memory_order::acq_rel);
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
        auto store = [](std::atomic_ref<T> const& x, T const& old_val, T const& new_val) {
          auto r = x.compare_exchange_strong(
              const_cast<Unqualified&>(old_val),
              const_cast<Unqualified const&>(new_val),
              std::memory_order::seq_cst,
              std::memory_order::relaxed);
          assert(r);
        };
        auto load = [](std::atomic_ref<T> const& x) {
          auto val = x.load(std::memory_order::relaxed);
          while (!x.compare_exchange_strong(val, val, std::memory_order::seq_cst, std::memory_order::relaxed)) {
          }
          return val;
        };
        test_seq_cst<T>(store, load);

        auto store_one_arg = [](std::atomic_ref<T> const& x, T const& old_val, T const& new_val) {
          auto r = x.compare_exchange_strong(
              const_cast<Unqualified&>(old_val), const_cast<Unqualified const&>(new_val), std::memory_order::seq_cst);
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
        auto store = [](std::atomic_ref<T> const& x, T const&, T const& new_val) {
          x.store(const_cast<Unqualified const&>(new_val), std::memory_order::release);
        };
        auto load = [](std::atomic_ref<T> const& x) {
          auto result = x.load(std::memory_order::relaxed);
          Unqualified unexpected(Unqualified(255));
          bool r =
              x.compare_exchange_strong(unexpected, unexpected, std::memory_order::relaxed, std::memory_order::acquire);
          assert(!r);
          return result;
        };
        test_acquire_release<T>(store, load);

        auto load_one_arg = [](std::atomic_ref<T> const& x) {
          auto result = x.load(std::memory_order::relaxed);
          Unqualified unexpected(Unqualified(255));
          bool r = x.compare_exchange_strong(unexpected, unexpected, std::memory_order::acquire);
          assert(!r);
          return result;
        };
        test_acquire_release<T>(store, load_one_arg);

        // acq_rel replaced by acquire
        auto load_one_arg_acq_rel = [](std::atomic_ref<T> const& x) {
          auto result = x.load(std::memory_order::relaxed);
          Unqualified unexpected(Unqualified(255));
          bool r = x.compare_exchange_strong(unexpected, unexpected, std::memory_order::acq_rel);
          assert(!r);
          return result;
        };
        test_acquire_release<T>(store, load_one_arg_acq_rel);
      }

      // failure memory_order::seq_cst
      {
        auto store = [](std::atomic_ref<T> const& x, T const&, T const& new_val) {
          x.store(const_cast<Unqualified const&>(new_val), std::memory_order::seq_cst);
        };
        auto load = [](std::atomic_ref<T> const& x) {
          auto result = x.load(std::memory_order::relaxed);
          Unqualified unexpected(Unqualified(255));
          bool r =
              x.compare_exchange_strong(unexpected, unexpected, std::memory_order::relaxed, std::memory_order::seq_cst);
          assert(!r);
          return result;
        };
        test_seq_cst<T>(store, load);
      }
    }
  }
};

template <template <class...> class F>
struct CallWithCVQualifiers {
  template <class T>
  struct apply {
    void operator()() const {
      F<T>()();
      F<T const>()();
      if constexpr (std::atomic_ref<T>::is_always_lock_free) {
        F<T volatile>()();
        F<T const volatile>()();
      }
    }
  };
};

int main(int, char**) {
  TestEachAtomicType<CallWithCVQualifiers<TestCompareExchangeStrong>::apply>()();
  return 0;
}
