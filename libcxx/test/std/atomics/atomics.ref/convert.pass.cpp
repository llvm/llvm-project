//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

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
    T x(T(1));

    T copy = x;
    std::atomic_ref<T> const a(copy);

    T converted = a;
    assert(converted == x);

    ASSERT_NOEXCEPT(T(a));
    static_assert(std::is_nothrow_convertible_v<std::atomic_ref<T>, T>);

    auto store = [](std::atomic_ref<T> const& y, T, T new_val) { y.store(new_val); };
    auto load  = [](std::atomic_ref<T> const& y) { return static_cast<T>(y); };
    test_seq_cst<T>(store, load);
  }
};

void test() {
  TestEachIntegralType<TestConvert>()();

  TestEachFloatingPointType<TestConvert>()();

  TestEachPointerType<TestConvert>()();

  TestConvert<UserAtomicType>()();
  TestConvert<LargeUserAtomicType>()();
}

int main(int, char**) {
  test();
  return 0;
}
