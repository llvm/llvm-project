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

  // Basic sanity check: trace should capture current frame
  assert(!trace.empty());
  assert(trace.size() > 0);

  // Check entry formatting logic
  std::stacktrace_entry entry = trace[0];
  std::string desc = std::to_string(entry);
  (void)desc;

  return 0;
}
