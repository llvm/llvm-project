//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

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
  std::stacktrace_entry entry_t6;
  // "Returns: false if and only if *this is empty."
  assert(!entry_t6);
  // Now set addr to something nonzero
  *(uintptr_t*)(&entry_t6) = uintptr_t(&main);
  assert(entry_t6.native_handle() == uintptr_t(&main));
  assert(entry_t6);

  return 0;
}
