//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// <mutex>

// GCC doesn't have thread safety attributes
// UNSUPPORTED: gcc

// ADDITIONAL_COMPILE_FLAGS: -Wthread-safety -Wthread-safety-negative

#include <mutex>

#include "test_macros.h"

std::mutex m;
int foo __attribute__((guarded_by(m)));

void increment() __attribute__((requires_negative_capability(m))) {
  m.lock();
  foo++;
  m.unlock();
}

void test() __attribute__((requires_negative_capability(m))) {
  increment();
}

int main(int, char**) {
  return 0;
}
