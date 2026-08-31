//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: no-localization
// UNSUPPORTED: availability-stacktrace-missing

// (19.6.4.6) Non-member functions [stacktrace.basic.nonmem]
//
//   string to_string(const stacktrace_entry& f);
//
//   template<class Allocator>
//     string to_string(const basic_stacktrace<Allocator>& st);

#include <cassert>
#include <stacktrace>

int main(int, char**) {
  auto trace = std::stacktrace::current();
  assert(!std::to_string(trace).empty());

  auto entry = trace[0];
  assert(!std::to_string(entry).empty());

  return 0;
}
