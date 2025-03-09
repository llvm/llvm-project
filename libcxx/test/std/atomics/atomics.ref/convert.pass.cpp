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

template <typename U>
struct TestConvert {
  void operator()() const {
    static_assert(std::is_nothrow_convertible_v<std::atomic_ref<U>, U>);
    do_test<U>();
    do_test_atomic<U>();
    static_assert(std::is_nothrow_convertible_v<std::atomic_ref<U const>, U>);
    do_test<U const>();
    if constexpr (std::atomic_ref<U>::is_always_lock_free) {
      static_assert(std::is_nothrow_convertible_v<std::atomic_ref<U volatile>, U>);
      do_test<U volatile>();
      do_test_atomic<U volatile>();
      static_assert(std::is_nothrow_convertible_v<std::atomic_ref<U const volatile>, U>);
      do_test<U const volatile>();
    }
  }

  template <class T>
  void do_test() const {
    T x(T(1));

    T copy = const_cast<std::remove_cv_t<T> const&>(x);
    std::atomic_ref<T> const a(copy);

    std::remove_cv_t<T> converted = a;
    assert(converted == const_cast<std::remove_cv_t<T> const&>(x));

    ASSERT_NOEXCEPT(T(a));
  }

  template <class T>
  void do_test_atomic() const {
    auto store = [](std::atomic_ref<T> const& y, T const&, T const& new_val) {
      y.store(const_cast<std::remove_cv_t<T> const&>(new_val));
    };
    auto load = [](std::atomic_ref<T> const& y) { return static_cast<std::remove_cv_t<T>>(y); };
    test_seq_cst<T>(store, load);
  }
};

int main(int, char**) {
  TestEachAtomicType<TestConvert>()();
  return 0;
}
