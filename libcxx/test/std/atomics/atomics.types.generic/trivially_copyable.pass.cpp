//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <atomic>

// template <class T>
// struct atomic;

// Make sure atomic<TriviallyCopyable> can be instantiated.

#include <atomic>
#include <new>
#include <cassert>
#include <thread> // for thread_id
#include <chrono> // for nanoseconds

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
  test(std::this_thread::get_id());
  test(std::chrono::nanoseconds(2));

  return 0;
}
