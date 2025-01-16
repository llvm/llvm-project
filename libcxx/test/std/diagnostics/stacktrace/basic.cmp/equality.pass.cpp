//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// XFAIL: availability-stacktrace-missing

// (19.6.4.4) Comparisons [stacktrace.basic.cmp]
// template<class Allocator2>
// friend bool operator==(const basic_stacktrace& x,
//                         const basic_stacktrace<Allocator2>& y) noexcept;

#include <cassert>
#include <stacktrace>

#include "test_macros.h"

// Disable TCO for calls into, and out from, the annotated function.
#define STACKTRACE_AVOID_OPT TEST_NO_TAIL_CALLS_IN TEST_NO_TAIL_CALLS_OUT TEST_NOINLINE

// Some non-inlinable functions to help contrive different stacktraces:
// main calls the "middle" funcs, and those both call "top".
// We'll consider main the "bottom" func, even though there are other functions
// like `_start` which call main; those are trimmed via `max_depth` argument.

STACKTRACE_AVOID_OPT std::stacktrace top(size_t skip, size_t depth) { return std::stacktrace::current(skip, depth); }
STACKTRACE_AVOID_OPT std::stacktrace middle1(size_t skip, size_t depth) { return top(skip, depth); }
STACKTRACE_AVOID_OPT std::stacktrace middle2(size_t skip, size_t depth) { return top(skip, depth); }

STACKTRACE_AVOID_OPT int main(int, char**) {
  // Collect a few different stacktraces and test `operator==` and `operator!=`.

  std::stacktrace st0;                 // default-initializable empty stacktrace
  static_assert(noexcept(st0 == st0)); // verify noexcept-ness
  static_assert(noexcept(st0 != st0)); // verify noexcept-ness
  assert(st0 == st0);                  // trivial: self-equality

  std::stacktrace st1a = top(0, 2);     // st1a = [top, main]
  assert(st1a == st1a);                 // trivial: self-equality
  assert(st1a != st0);                  //
  std::stacktrace st2a = middle1(0, 3); // st2a = [top, middle1, main]
  assert(st2a == st2a);                 //
  assert(st1a != st2a);                 //
  std::stacktrace st2b = middle2(0, 3); // st2b = [top, middle2, main]
  assert(st2b == st2b);                 //
  assert(st2a != st2b);                 //

  // Verify two equivalent stacktrace instances are equal, even if not "same".
  // For both, we'll take only two entries, which should be equivalent.
  std::stacktrace st3a = middle1(0, 2); // st3a = [top, middle1]
  std::stacktrace st3b = middle1(0, 2); // st3b = [top, middle1]
  assert(st3a == st3b);

  return 0;
}
