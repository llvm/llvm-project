//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// bool compare_exchange_strong(weak_ptr<T>&, weak_ptr<T>, memory_order, memory_order) noexcept;
// bool compare_exchange_strong(weak_ptr<T>&, weak_ptr<T>, memory_order = memory_order_seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_compare_exchange_strong_weakptr() {
  auto sp1             = SpValues<T>::state_a();
  auto sp2             = SpValues<T>::state_b();
  std::weak_ptr<T> wp1 = sp1;
  std::weak_ptr<T> wp2 = sp2;

  std::atomic<std::weak_ptr<T>> a((std::weak_ptr<T>(wp1)));

  {
    std::weak_ptr<T> expected            = wp1;
    std::same_as<bool> decltype(auto) ok = a.compare_exchange_strong(expected, std::weak_ptr<T>(wp2));
    assert(ok);
    {
      auto locked = a.load().lock();
      assert(locked && *locked == *sp2);
    }
    static_assert(noexcept(a.compare_exchange_strong(expected, std::weak_ptr<T>(wp2))));
  }

  {
    std::weak_ptr<T> expected = wp1;
    bool ok                   = a.compare_exchange_strong(expected, std::weak_ptr<T>(wp1), std::memory_order_seq_cst);
    assert(!ok);
    auto locked = expected.lock();
    assert(locked && *locked == *sp2);
  }

  {
    std::weak_ptr<T> expected = wp2;
    auto sp3                  = SpValues<T>::state_c();
    bool ok                   = a.compare_exchange_strong(
        expected, std::weak_ptr<T>(sp3), std::memory_order_release, std::memory_order_relaxed);
    assert(ok);
    auto locked = a.load().lock();
    assert(locked && *locked == *sp3);
  }
}

template <class T>
struct TestCompareExchangeStrongWeak {
  void operator()() const { test_compare_exchange_strong_weakptr<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestCompareExchangeStrongWeak>();
  return 0;
}
