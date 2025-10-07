//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <atomic> functions are marked [[nodiscard]]

// clang-format off

#include <atomic>

#include "atomic_helpers.h"
#include "test_macros.h"

template <typename T>
struct TestAtomicRef {
  void operator()() const {
    T x(T(1));
    const std::atomic_ref<T> a(x);

#if TEST_STD_VER >= 26
  a.address(); // expected-warning 4 {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
  }
};

void test() {
  TestAtomicRef<UserAtomicType>()();
  TestAtomicRef<int>()();
  TestAtomicRef<float>()();
  TestAtomicRef<char*>()();
}
