//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// template <class T>
// struct atomic;

// Make sure atomic<TriviallyCopyable> can be instantiated.

#include <atomic>
#include <new>
#include <cassert>
#include <chrono> // for nanoseconds

#include "test_macros.h"

#ifndef TEST_HAS_NO_THREADS
#  include <thread> // for thread_id
#endif

struct TriviallyCopyable {
  explicit TriviallyCopyable(int i) : i_(i) { }
  int i_;
};

template <class T>
void test(T t) {
  std::atomic<T> t0(t);
}

int main(int, char**) {
  test(TriviallyCopyable(42));
  test(std::chrono::nanoseconds(2));
#ifndef TEST_HAS_NO_THREADS
  test(std::this_thread::get_id());
#endif

  return 0;
}
