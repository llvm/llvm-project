//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <stacktrace>

#include <stacktrace>
#include <cassert>
#include "test_macros.h"

int main(int, char**) {
  std::stacktrace trace = std::stacktrace::current();
  if (trace.size() >= 2) {
    std::stacktrace_entry e1 = trace[0];
    std::stacktrace_entry e2 = trace[1];

    assert(e1 == e1);
    assert(!(e1 != e1));
  }
  return 0;
}
