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
  (19.6.3.3) Observers [stacktrace.entry.obs]

namespace std {
  class stacktrace_entry {
    // [stacktrace.entry.obs], observers
    constexpr explicit operator bool() const noexcept;
*/

#include <cassert>
#include <stacktrace>

int main(int, char**) {
  std::stacktrace_entry entry;

  static_assert(noexcept(bool(entry)));

  // "Returns: false if and only if *this is empty."
  assert(!entry);

  std::stacktrace trace = std::stacktrace::current();
  assert(trace[0]);

  return 0;
}
