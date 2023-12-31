//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <atomic>

// static constexpr bool is_always_lock_free;
// bool is_lock_free() const noexcept;

#include <atomic>
#include <cassert>

#include "test_macros.h"

template <typename T>
void checkAlwaysLockFree(std::atomic_ref<T> a) {
  if (std::atomic_ref<T>::is_always_lock_free) {
    assert(a.is_lock_free());
  }
  ASSERT_NOEXCEPT(a.is_lock_free());
}

void test() {
  int i = 0;
  checkAlwaysLockFree(std::atomic_ref<int>(i));

  float f = 0.f;
  checkAlwaysLockFree(std::atomic_ref<float>(f));

  int* p = &i;
  checkAlwaysLockFree(std::atomic_ref<int*>(p));

  struct X {
  } x;
  checkAlwaysLockFree(std::atomic_ref<X>(x));
}

int main(int, char**) {
  test();
  return 0;
}
