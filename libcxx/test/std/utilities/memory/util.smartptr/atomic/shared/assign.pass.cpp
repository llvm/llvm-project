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
  auto p = libcxx_atomic_smart_ptr_test::SpValues<T>::state_a();
  a      = std::shared_ptr<T>(p);
  assert(a.load().get() == p.get());
  assert(*a.load() == *p);
  a = nullptr;
  assert(!a.load());
  ASSERT_NOEXCEPT(a = nullptr);
  ASSERT_NOEXCEPT(a = std::shared_ptr<T>(p));
}

int main(int, char**) {
#define LIBCXX_ATOMIC_SP_RUN_ASSIGN(T) test_assign<T>();
  LIBCXX_ATOMIC_SP_FOR_ALL_RUNTIME_TYPES(LIBCXX_ATOMIC_SP_RUN_ASSIGN)
#undef LIBCXX_ATOMIC_SP_RUN_ASSIGN
  return 0;
}
