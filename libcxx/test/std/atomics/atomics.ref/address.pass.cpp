//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constexpr T* address() const noexcept;

#include <atomic>
#include <cassert>
#include <concepts>
#include <memory>

#include "atomic_helpers.h"
#include "test_macros.h"

template <typename T>
struct TestAddress {
  void operator()() const {
    T x(T(1));
    const std::atomic_ref<T> a(x);

    std::same_as<T*> decltype(auto) p = a.address();
    assert(std::addressof(x) == p);

    static_assert(noexcept((a.address())));
  }
};

int main(int, char**) {
  TestEachAtomicType<TestAddress>()();

  return 0;
}
