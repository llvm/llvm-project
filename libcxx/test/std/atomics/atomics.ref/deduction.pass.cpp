//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <atomic>

#include <atomic>
#include <type_traits>

void test() {
  int i = 0;
  std::atomic_ref a0(i);
  static_assert(std::is_same_v<decltype(a0), std::atomic_ref<int>>);

  float f = 0.f;
  std::atomic_ref a1(f);
  static_assert(std::is_same_v<decltype(a1), std::atomic_ref<float>>);

  int* p = &i;
  std::atomic_ref a2(p);
  static_assert(std::is_same_v<decltype(a2), std::atomic_ref<int*>>);

  struct X {
  } x;
  std::atomic_ref a3(x);
  static_assert(std::is_same_v<decltype(a3), std::atomic_ref<X>>);
}

int main(int, char**) {
  test();
  return 0;
}
