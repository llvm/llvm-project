//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -O0 -g
// XFAIL: availability-stacktrace-missing

/*
  (19.6.6) Hash Support

  template<> struct hash<stacktrace_entry>;                                 [1]
  template<class Allocator> struct hash<basic_stacktrace<Allocator>>;       [2]

  The specializations are enabled ([unord.hash]).
*/

#include <cassert>
#include <functional>
#include <stacktrace>

int main(int, char**) {
  std::stacktrace trace; // initially empty
  auto hash_val_empty    = std::hash<std::stacktrace>()(trace);
  trace                  = /* reassign */ std::stacktrace::current();
  auto hash_val_nonempty = std::hash<std::stacktrace>()(trace);
  assert(hash_val_empty != hash_val_nonempty);

  std::stacktrace_entry empty_entry;
  assert(std::hash<std::stacktrace_entry>()(empty_entry) == 0);
  auto nonempty_entry = trace[0];
  assert(std::hash<std::stacktrace_entry>()(nonempty_entry) != 0);

  return 0;
}
