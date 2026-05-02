//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// operator shared_ptr<T>() const noexcept;

#include <atomic>
#include <cassert>
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_convert() {
  auto p = SpValues<T>::state_a();
  const std::atomic<std::shared_ptr<T>> a((std::shared_ptr<T>(p)));

  std::shared_ptr<T> s = a;
  assert(s.get() == p.get());
  assert(*s == *p);

  std::same_as<std::shared_ptr<T>> decltype(auto) s2 = static_cast<std::shared_ptr<T>>(a);
  assert(s2.get() == p.get());

  static_assert(noexcept(static_cast<std::shared_ptr<T>>(a)));
}

template <class T>
struct TestConvert {
  void operator()() const { test_convert<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestConvert>();
  return 0;
}
