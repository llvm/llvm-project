//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// shared_ptr<T> exchange(shared_ptr<T> desired, memory_order order = memory_order::seq_cst) noexcept;

#include <atomic>
#include <cassert>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_exchange() {
  auto p1 = SpValues<T>::state_a();
  auto p2 = SpValues<T>::state_b();
  std::atomic<std::shared_ptr<T>> a((std::shared_ptr<T>(p1)));

  std::same_as<std::shared_ptr<T>> decltype(auto) out = a.exchange(std::shared_ptr<T>(p2));
  assert(*out == *p1);
  assert(out.get() == p1.get());
  assert(*a.load() == *p2);

  out = a.exchange(nullptr, std::memory_order_seq_cst);
  assert(*out == *p2);
  assert(!a.load());

  static_assert(noexcept(a.exchange(nullptr)));
}

template <class T>
struct TestExchange {
  void operator()() const { test_exchange<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestExchange>();
  return 0;
}
