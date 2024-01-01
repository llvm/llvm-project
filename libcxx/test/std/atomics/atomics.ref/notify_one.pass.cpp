//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: *

// void notify_one() const noexcept;

#include <atomic>
#include <cassert>
#include <thread>
#include <type_traits>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

template <typename T>
void test_notify_one() {
  T x(T(1));
  std::atomic_ref<T> a(x);

  std::thread t = support::make_test_thread([&]() {
    a.store(T(3));
    a.notify_one();
  });
  a.wait(T(1));
  assert(a.load() == T(3));
  t.join();
  ASSERT_NOEXCEPT(a.notify_one());
}

void test() {
  test_notify_one<int>();

  test_notify_one<float>();

  test_notify_one<int*>();

  struct X {
    int i;
    X(int ii) noexcept : i(ii) {}
    bool operator==(X o) const { return i == o.i; }
  };
  test_notify_one<X>();
}

int main(int, char**) {
  test();
  return 0;
}
