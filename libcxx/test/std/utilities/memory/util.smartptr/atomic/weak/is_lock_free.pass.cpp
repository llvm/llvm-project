//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// bool is_lock_free() const noexcept;

#include <atomic>
#include <concepts> // needed on Armv7/Armv8 with -fmodules
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void check(const std::atomic<std::weak_ptr<T>>& awp) noexcept {
  static_assert(!std::atomic<std::weak_ptr<T>>::is_always_lock_free);
  std::same_as<bool> decltype(auto) lf = awp.is_lock_free();
  (void)lf;
  static_assert(noexcept(awp.is_lock_free()));
}

template <class T>
struct TestIsLockFreeWeak {
  void operator()() const { check<T>(std::atomic<std::weak_ptr<T>>()); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestIsLockFreeWeak>();
  return 0;
}
