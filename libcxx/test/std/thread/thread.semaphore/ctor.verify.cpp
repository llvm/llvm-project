//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <semaphore>

#include <semaphore>

void not_positive() {
  std::counting_semaphore<-1> s(2); // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed{{.*}}The least maximum value must be a positive number}}
  (void)s;
}
