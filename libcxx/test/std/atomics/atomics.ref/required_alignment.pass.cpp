//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// static constexpr size_t required_alignment;

#include <atomic>

template <typename T>
void checkRequiredAlignment() {
  static_assert(std::atomic_ref<T>::required_alignment >= alignof(T));
}

void test() {
  checkRequiredAlignment<int>();
  checkRequiredAlignment<float>();
  checkRequiredAlignment<int*>();
  struct Empty {};
  checkRequiredAlignment<Empty>();
  struct Trivial {
    int a;
  };
  checkRequiredAlignment<Trivial>();
}

int main(int, char**) {
  test();
  return 0;
}
