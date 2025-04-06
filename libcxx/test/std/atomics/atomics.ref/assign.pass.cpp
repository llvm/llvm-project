//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics
// XFAIL: !has-1024-bit-atomics

// T operator=(T) const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
struct TestAssign {
  void operator()() const {
    using Unqualified = std::remove_cv_t<T>;
    static_assert(std::is_assignable_v<std::atomic_ref<T>, Unqualified> == !std::is_const_v<T>);

    if constexpr (!std::is_const_v<T>) {
      {
        T x(Unqualified(1));
        std::atomic_ref<T> const a(x);

        std::same_as<Unqualified> decltype(auto) y = (a = Unqualified(2));
        assert(y == Unqualified(2));
        assert(const_cast<Unqualified const&>(x) == Unqualified(2));

        ASSERT_NOEXCEPT(a = Unqualified(0));
        static_assert(std::is_nothrow_assignable_v<std::atomic_ref<T>, Unqualified>);
        static_assert(!std::is_copy_assignable_v<std::atomic_ref<T>>);
      }

      {
        auto assign = [](std::atomic_ref<T> const& y, T const&, T const& new_val) {
          y = const_cast<Unqualified const&>(new_val);
        };
        auto load = [](std::atomic_ref<T> const& y) { return y.load(); };
        test_seq_cst<T>(assign, load);
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
  TestEachAtomicType<CallWithCVQualifiers<TestAssign>::apply>()();
  return 0;
}
