//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: !has-64-bit-atomics
// XFAIL: !has-1024-bit-atomics

// operator T() const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_helper.h"
#include "test_macros.h"

template <typename T>
struct TestConvert {
  void operator()() const {
    using Unqualified = std::remove_cv_t<T>;
    static_assert(std::is_nothrow_convertible_v<std::atomic_ref<T>, Unqualified>);

    {
      T x(Unqualified(1));

      T copy = const_cast<Unqualified const&>(x);
      std::atomic_ref<T> const a(copy);

      Unqualified converted = a;
      assert(converted == const_cast<Unqualified const&>(x));

      ASSERT_NOEXCEPT(T(a));
    }

    if constexpr (!std::is_const_v<T>) { // FIXME
      auto store = [](std::atomic_ref<T> const& y, T const&, T const& new_val) {
        y.store(const_cast<Unqualified const&>(new_val));
      };
      auto load = [](std::atomic_ref<T> const& y) { return static_cast<Unqualified>(y); };
      test_seq_cst<T>(store, load);
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
  TestEachAtomicType<CallWithCVQualifiers<TestConvert>::apply>()();
  return 0;
}
