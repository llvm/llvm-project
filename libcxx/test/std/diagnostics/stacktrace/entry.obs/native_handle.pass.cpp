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
    constexpr native_handle_type native_handle() const noexcept;
*/

#include <cassert>
#include <cstdint>
#include <stacktrace>

int main(int, char**) noexcept {
  std::stacktrace_entry entry;
  static_assert(noexcept(entry.native_handle()));
  assert(entry.native_handle() == 0);

  std::stacktrace trace = std::stacktrace::current();
  assert(trace[0].native_handle() != 0);
  assert(trace[0].native_handle() > uintptr_t(&main));

  return 0;
}
