//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

// template <class T, class Container = deque<T>>
// class queue

#include <queue>

void test() {
  // expected-error-re@*:* {{static assertion failed{{.*}}{{(The stored elements type must match the underlying container's value_type\.)?}}}}
  std::queue<double, std::deque<int> > qu;
}
