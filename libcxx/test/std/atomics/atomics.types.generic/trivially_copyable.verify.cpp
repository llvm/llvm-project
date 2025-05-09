//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// <atomic>

// template <class T>
// struct atomic;

#include <atomic>

struct NotTriviallyCopyable {
  explicit NotTriviallyCopyable(int i) : i_(i) {}
  NotTriviallyCopyable(const NotTriviallyCopyable& rhs) : i_(rhs.i_) {}
  int i_;
};

void f() {
  NotTriviallyCopyable x(42);
  std::atomic<NotTriviallyCopyable> a(
      x); // expected-error@*:* {{std::atomic<T> requires that 'T' be a trivially copyable type}}
}
