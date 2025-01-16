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
// friend strong_ordering operator<=>(const basic_stacktrace& x,
//                                    const basic_stacktrace<Allocator2>& y) noexcept;
//
// Returns: x.size() <=> y.size() if x.size() != y.size();
// lexicographical_compare_three_way(x.begin(), x.end(), y.begin(), y.end()) otherwise.

#include <cassert>
#include <stacktrace>

#include "test_macros.h"

// Disable TCO for calls into, and out from, the annotated function.
#define STACKTRACE_AVOID_OPT TEST_NO_TAIL_CALLS_IN TEST_NO_TAIL_CALLS_OUT TEST_NOINLINE

// Some non-inlinable functions to help contrive different stacktraces:
// main calls the "middle" funcs, and those both call "top".
// We'll consider main the "bottom" func, even though there are other functions
// like `_start` which call main; those are trimmed via `max_depth` argument.

STACKTRACE_AVOID_OPT std::stacktrace top() { return std::stacktrace::current(); }
STACKTRACE_AVOID_OPT std::stacktrace middle1() { return top(); }
STACKTRACE_AVOID_OPT std::stacktrace middle2() { return top(); }

STACKTRACE_AVOID_OPT int main(int, char**) {
  // Collect a few different stacktraces and test `operator<=>`.

  auto st1a = top(); // [top, main, ...]
  auto st1b = st1a;

  static_assert(noexcept(st1a <=> st1b));

  assert(st1a == st1b);
  auto st2a = middle1(); // [top, middle1, main, ...]
  assert(st1a != st2a);
  std::stacktrace empty; // []
  auto st2b = middle2(); // [top, middle2, main, ...]
  assert(st2a != st2b);

  // empty:  []
  // st1a:   [top, main, ...]
  // st1b:   [top, main, ...] (copy of st1a)
  // st2a:   [top, middle1, main:X, ...]
  // st2b:   [top, middle2, main:Y, ...], Y > X

  assert(std::strong_ordering::equal == empty <=> empty);
  assert(std::strong_ordering::less == empty <=> st1a);
  assert(std::strong_ordering::less == empty <=> st1b);
  assert(std::strong_ordering::less == empty <=> st2a);
  assert(std::strong_ordering::less == empty <=> st2b);

  assert(std::strong_ordering::greater == st1a <=> empty);
  assert(std::strong_ordering::equal == st1a <=> st1a);
  assert(std::strong_ordering::equal == st1a <=> st1b);
  assert(std::strong_ordering::less == st1a <=> st2a);
  assert(std::strong_ordering::less == st1a <=> st2b);

  assert(std::strong_ordering::greater == st1b <=> empty);
  assert(std::strong_ordering::equal == st1b <=> st1a);
  assert(std::strong_ordering::equal == st1b <=> st1b);
  assert(std::strong_ordering::less == st1b <=> st2a);
  assert(std::strong_ordering::less == st1b <=> st2b);

  assert(std::strong_ordering::greater == st2a <=> empty);
  assert(std::strong_ordering::greater == st2a <=> st1a);
  assert(std::strong_ordering::greater == st2a <=> st1b);
  assert(std::strong_ordering::equal == st2a <=> st2a);
  assert(std::strong_ordering::less == st2a <=> st2b);

  assert(std::strong_ordering::greater == st2b <=> empty);
  assert(std::strong_ordering::greater == st2b <=> st1a);
  assert(std::strong_ordering::greater == st2b <=> st1b);
  assert(std::strong_ordering::greater == st2b <=> st2a);
  assert(std::strong_ordering::equal == st2b <=> st2b);

  return 0;
}
