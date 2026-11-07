//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// void store(weak_ptr<T> desired, memory_order order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_store_weak() {
  auto sp1             = SpValues<T>::state_a();
  auto sp2             = SpValues<T>::state_b();
  std::weak_ptr<T> wp1 = sp1;
  std::weak_ptr<T> wp2 = sp2;

  std::atomic<std::weak_ptr<T>> a;
  a.store(std::weak_ptr<T>(wp1));
  {
    auto locked = a.load().lock();
    assert(locked && *locked == *sp1);
  }

  a.store(std::weak_ptr<T>(wp2), std::memory_order_relaxed);
  {
    auto locked = a.load().lock();
    assert(locked && *locked == *sp2);
  }

  static_assert(noexcept(a.store(std::weak_ptr<T>(wp1))));
}

template <class T>
struct TestStoreWeak {
  void operator()() const { test_store_weak<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestStoreWeak>();
  return 0;
}
