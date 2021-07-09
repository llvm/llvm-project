//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03

// <atomic>

// constexpr atomic<T>::atomic(T value)

#include <atomic>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "atomic_helpers.h"

struct UserType {
  int i;

  UserType() noexcept {}
  constexpr explicit UserType(int d) noexcept : i(d) {}

  friend bool operator==(const UserType& x, const UserType& y) { return x.i == y.i; }
};

template <class Tp>
struct TestFunc {
  void operator()() const {
    typedef std::atomic<Tp> Atomic;
    constexpr Tp t(42);
    {
      constexpr Atomic a(t);
      assert(a == t);
    }
    {
      constexpr Atomic a{t};
      assert(a == t);
    }
    {
      constexpr Atomic a = ATOMIC_VAR_INIT(t);
      assert(a == t);
    }
  }
};

int main(int, char**) {
  TestFunc<UserType>()();
  TestEachIntegralType<TestFunc>()();

  return 0;
}
