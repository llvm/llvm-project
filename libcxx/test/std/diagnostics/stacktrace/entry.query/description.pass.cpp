//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

/*
    (19.6.3.4) Query [stacktrace.entry.query]

namespace std {
  class stacktrace_entry {
  public:
    // [stacktrace.entry.query], query
    string description() const;
*/

#include <cassert>
#include <stacktrace>
#include <string>

int main(int, char**) {
  std::stacktrace_entry entry;
  assert(entry.description().empty());

  return 0;
}
