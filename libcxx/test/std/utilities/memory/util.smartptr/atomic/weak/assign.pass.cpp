//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// void operator=(weak_ptr<T>) noexcept;

#include <atomic>
#include <cassert>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_assign_weak() {
  auto sp1             = SpValues<T>::state_a();
  auto sp2             = SpValues<T>::state_b();
  std::weak_ptr<T> wp1 = sp1;
  std::weak_ptr<T> wp2 = sp2;

  std::atomic<std::weak_ptr<T>> a;
  a = std::weak_ptr<T>(wp1);
  {
    auto locked = a.load().lock();
    assert(locked && *locked == *sp1);
  }

  a = std::weak_ptr<T>(wp2);
  {
    auto locked = a.load().lock();
    assert(locked && *locked == *sp2);
  }

  static_assert(noexcept(a = std::weak_ptr<T>(wp1)));
}

template <class T>
struct TestAssignWeak {
  void operator()() const { test_assign_weak<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestAssignWeak>();
  return 0;
}
