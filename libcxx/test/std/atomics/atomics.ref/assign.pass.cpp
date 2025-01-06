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
    {
      T x(T(1));
      std::atomic_ref<T> const a(x);

      std::same_as<T> decltype(auto) y = (a = T(2));
      assert(y == T(2));
      assert(x == T(2));

      ASSERT_NOEXCEPT(a = T(0));
      static_assert(std::is_nothrow_assignable_v<std::atomic_ref<T>, T>);

      static_assert(!std::is_copy_assignable_v<std::atomic_ref<T>>);
    }

    {
      auto assign = [](std::atomic_ref<T> const& y, T, T new_val) { y = new_val; };
      auto load   = [](std::atomic_ref<T> const& y) { return y.load(); };
      test_seq_cst<T>(assign, load);
    }
  }
};

int main(int, char**) {
  TestEachAtomicType<TestAssign>()();
  return 0;
}
