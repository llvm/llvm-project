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

template <typename U>
struct TestAssign {
  void operator()() const {
    static_assert(std::is_nothrow_assignable_v<std::atomic_ref<U>, U>);
    do_test<U>();
    do_test_atomic<U>();
    static_assert(!std::is_assignable_v<std::atomic_ref<U const>, U>);
    if constexpr (std::atomic_ref<U>::is_always_lock_free) {
      static_assert(std::is_nothrow_assignable_v<std::atomic_ref<U volatile>, U>);
      do_test<U volatile>();
      do_test_atomic<U volatile>();
      static_assert(!std::is_assignable_v<std::atomic_ref<U const volatile>, U>);
    }
  }

  template <typename T>
  void do_test() const {
    T x(T(1));
    std::atomic_ref<T> const a(x);

    std::same_as<std::remove_cv_t<T>> decltype(auto) y = (a = T(2));
    assert(y == std::remove_cv_t<T>(2));
    assert(const_cast<std::remove_cv_t<T> const&>(x) == std::remove_cv_t<T>(2));

    ASSERT_NOEXCEPT(a = T(0));

    static_assert(!std::is_copy_assignable_v<std::atomic_ref<T>>);
  }

  template <typename T>
  void do_test_atomic() const {
    {
      auto assign = [](std::atomic_ref<T> const& y, T const&, T const& new_val) {
        y = const_cast<std::remove_cv_t<T> const&>(new_val);
      };
      auto load = [](std::atomic_ref<T> const& y) { return y.load(); };
      test_seq_cst<T>(assign, load);
    }
  }
};

int main(int, char**) {
  TestEachAtomicType<TestAssign>()();
  return 0;
}
