//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <atomic>

// explicit atomic_ref(T&);

#include <atomic>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_macros.h"

template <typename T>
struct TestDeduction {
  void operator()() const {
    do_test<T>();
    do_test<T const>();
    if constexpr (std::atomic_ref<T>::is_always_lock_free) {
      do_test<T volatile>();
      do_test<T const volatile>();
    }
  }
  template <class U>
  void do_test() const {
    U x(U(0));
    std::atomic_ref a(x);
    ASSERT_SAME_TYPE(decltype(a), std::atomic_ref<U>);
  }
};

int main(int, char**) {
  TestEachAtomicType<TestDeduction>()();
  return 0;
}
