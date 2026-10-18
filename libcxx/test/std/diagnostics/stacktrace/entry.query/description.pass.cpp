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

#include "test_macros.h"

TEST_NOINLINE std::stacktrace f() { return std::stacktrace::current(); }

int main(int, char**) {
  std::stacktrace_entry entry;
  assert(entry.description().empty());

  // description() is inherently best-effort: on POSIX it relies on `dladdr` finding the address
  // in the running image's *dynamic* symbol table, which by default does not include a plain
  // executable's own internal symbols unless it's linked with `-rdynamic` (see basic.cons/
  // current.pass.cpp's similar comment about not being able to rely on symbol resolution in the
  // test environment). So just exercise the real-capture path here without asserting content.
  entry = f()[0];
  (void)entry.description();

  return 0;
}
