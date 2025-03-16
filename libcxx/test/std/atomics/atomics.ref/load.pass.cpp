//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics
// XFAIL: !has-1024-bit-atomics

// T load(memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <concepts>
#include <cassert>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
struct TestLoad {
  void operator()() const {
    using Unqualified = std::remove_cv_t<T>;

    T x(Unqualified(1));
    std::atomic_ref<T> const a(x);

    {
      std::same_as<Unqualified> decltype(auto) y = a.load();
      assert(y == Unqualified(1));
      ASSERT_NOEXCEPT(a.load());
    }

    {
      std::same_as<Unqualified> decltype(auto) y = a.load(std::memory_order_seq_cst);
      assert(y == Unqualified(1));
      ASSERT_NOEXCEPT(a.load(std::memory_order_seq_cst));
    }

    if constexpr (!std::is_const_v<T>) { // FIXME
      // memory_order::seq_cst
      {
        auto store = [](std::atomic_ref<T> const& y, T const&, T const& new_val) {
          y.store(const_cast<Unqualified const&>(new_val));
        };
        auto load_no_arg     = [](std::atomic_ref<T> const& y) { return y.load(); };
        auto load_with_order = [](std::atomic_ref<T> const& y) { return y.load(std::memory_order::seq_cst); };
        test_seq_cst<T>(store, load_no_arg);
        test_seq_cst<T>(store, load_with_order);
      }

      // memory_order::release
      {
        auto store = [](std::atomic_ref<T> const& y, T const&, T const& new_val) {
          y.store(const_cast<Unqualified const&>(new_val), std::memory_order::release);
        };
        auto load = [](std::atomic_ref<T> const& y) { return y.load(std::memory_order::acquire); };
        test_acquire_release<T>(store, load);
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
  TestEachAtomicType<CallWithCVQualifiers<TestLoad>::apply>()();
  return 0;
}
