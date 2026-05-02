//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// static constexpr bool is_always_lock_free;
// bool is_lock_free() const noexcept;

#include <atomic>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void check() {
  using A = std::atomic<std::weak_ptr<T>>;

  static_assert(std::same_as<decltype(A::is_always_lock_free), const bool>);
  static_assert(A::is_always_lock_free == false);

  const A awp;
  std::same_as<bool> decltype(auto) lf = awp.is_lock_free();
  (void)lf;
  static_assert(noexcept(A::is_always_lock_free));
  static_assert(noexcept(awp.is_lock_free()));
}

template <class T>
struct TestIsAlwaysLockFreeWeak {
  void operator()() const { check<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestIsAlwaysLockFreeWeak>();
  return 0;
}
