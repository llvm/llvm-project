//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// UNSUPPORTED: no-threads

// operator weak_ptr<T>() const noexcept;

#include <atomic>
#include <cassert>
#include <concepts> // needed on Armv7/Armv8 with -fmodules
#include <memory>

#include "../atomic_smart_ptr_test_types.h"
#include "test_macros.h"

template <class T>
void test_convert_weak() {
  auto sp             = SpValues<T>::state_a();
  std::weak_ptr<T> wp = sp;
  const std::atomic<std::weak_ptr<T>> a((std::weak_ptr<T>(wp)));

  std::weak_ptr<T> w = a;
  auto locked        = w.lock();
  assert(locked && *locked == *sp);

  std::same_as<std::weak_ptr<T>> decltype(auto) w2 = static_cast<std::weak_ptr<T>>(a);
  auto locked2                                     = w2.lock();
  assert(locked2 && *locked2 == *sp);

  static_assert(noexcept(static_cast<std::weak_ptr<T>>(a)));
}

template <class T>
struct TestConvertWeak {
  void operator()() const { test_convert_weak<T>(); }
};

int main(int, char**) {
  ForEachSmartPtrType{}.template operator()<TestConvertWeak>();
  return 0;
}
