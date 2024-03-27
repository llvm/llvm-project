//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// void store(T, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "atomic_helpers.h"
#include "test_macros.h"

template <typename T>
struct TestStore {
  void operator()() const {
    T x(T(1));
    std::atomic_ref<T> const a(x);

    a.store(T(2));
    assert(x == T(2));
    ASSERT_NOEXCEPT(a.store(T(1)));

    a.store(T(3), std::memory_order_seq_cst);
    assert(x == T(3));
    ASSERT_NOEXCEPT(a.store(T(0), std::memory_order_seq_cst));
  }
};

void test() {
  TestEachIntegralType<TestStore>()();

  TestEachFloatingPointType<TestStore>()();

  TestEachPointerType<TestStore>()();

  TestStore<UserAtomicType>()();
  TestStore<LargeUserAtomicType>()();
}

int main(int, char**) {
  test();
  return 0;
}
