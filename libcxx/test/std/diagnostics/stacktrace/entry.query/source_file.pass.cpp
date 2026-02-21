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
    string source_file() const;
*/

#include <cassert>
#include <stacktrace>
#include <string>

int main(int, char**) {
  std::stacktrace_entry entry;
  auto src = entry.source_file();
  assert(src.empty());

  entry = std::stacktrace::current()[0];
  src   = entry.source_file();
  assert(!src.empty());
  assert(src.find("source_file.pass.cpp") != std::string::npos);

  return 0;
}
