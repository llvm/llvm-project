//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// bool compare_exchange_weak(shared_ptr<T>&, shared_ptr<T>, memory_order, memory_order) noexcept;
// bool compare_exchange_weak(shared_ptr<T>&, shared_ptr<T>, memory_order = memory_order_seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_compare_exchange_weak() {
  auto p1 = SpValues<T>::state_a();
  auto p2 = SpValues<T>::state_b();
  std::atomic<std::shared_ptr<T>> a((std::shared_ptr<T>(p1)));

  {
    std::shared_ptr<T> expected = p1;
    bool ok                     = false;
    while (!ok)
      ok = a.compare_exchange_weak(expected, std::shared_ptr<T>(p2));
    assert(ok);
    assert(*a.load() == *p2);
    static_assert(noexcept(a.compare_exchange_weak(expected, std::shared_ptr<T>(p2))));
  }

  {
    std::shared_ptr<T> expected = p1;
    bool ok =
        a.compare_exchange_weak(expected, std::make_shared<T>(*SpValues<T>::state_c()), std::memory_order_seq_cst);
    assert(!ok);
    assert(*expected == *p2);
  }

  {
    std::shared_ptr<T> expected   = p2;
    const std::shared_ptr<T> next = SpValues<T>::state_c();
    bool ok                       = false;
    while (!ok)
      ok = a.compare_exchange_weak(
          expected, std::shared_ptr<T>(next), std::memory_order_release, std::memory_order_relaxed);
    assert(ok);
    assert(*a.load() == *next);
  }
}

template <class T>
struct TestCompareExchangeWeak {
  void operator()() const { test_compare_exchange_weak<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestCompareExchangeWeak>();
  return 0;
}
