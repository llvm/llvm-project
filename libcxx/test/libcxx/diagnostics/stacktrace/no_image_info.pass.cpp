//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// REQUIRES: availability-stacktrace-no-image-info

// Mirror image of the tests that are UNSUPPORTED on availability-stacktrace-no-image-info
// (e.g. std/diagnostics/stacktrace/entry.query/source_file.pass.cpp): those assume the platform
// can resolve captured addresses back to an image/source file, which doesn't hold here (no
// dynamic loader, no filesystem). This test runs *only* on such platforms, and checks the
// opposite: that stacktrace still works -- capture, indexing, comparison, hashing -- without
// crashing, even though every entry's file/line/description will come back empty.

#include <cassert>
#include <functional>
#include <stacktrace>

#include "test_macros.h"

TEST_NOINLINE std::stacktrace f() { return std::stacktrace::current(); }

int main(int, char**) {
  std::stacktrace st = f();

  assert(!st.empty());
  assert(st.size() > 0);

  for (std::stacktrace_entry const& entry : st) {
    assert(static_cast<bool>(entry));
    (void)entry.native_handle(); // just needs to not crash

    // No image info is available on this platform: these come back empty/0, not garbage.
    assert(entry.source_file().empty());
    assert(entry.source_line() == 0);
    assert(entry.description().empty());
  }

  (void)st.at(0);
  (void)st[0];

  std::stacktrace st2 = f();
  (void)(st == st2);
  (void)std::hash<std::stacktrace>{}(st);

  return 0;
}
