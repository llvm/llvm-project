//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// void operator=(shared_ptr<T>) noexcept;
// void operator=(nullptr_t) noexcept;

#include <atomic>
#include <cassert>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_assign() {
  std::atomic<std::shared_ptr<T>> a;
  auto p = SpValues<T>::state_a();
  a      = std::shared_ptr<T>(p);
  assert(a.load().get() == p.get());
  assert(*a.load() == *p);
  a = nullptr;
  assert(!a.load());
  static_assert(noexcept(a = nullptr));
  static_assert(noexcept(a = std::shared_ptr<T>(p)));
}

template <class T>
struct TestAssign {
  void operator()() const { test_assign<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestAssign>();
  return 0;
}
