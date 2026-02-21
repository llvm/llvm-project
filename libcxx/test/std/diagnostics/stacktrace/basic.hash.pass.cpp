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
  (19.6.6) Hash Support

  template<> struct hash<stacktrace_entry>;
  template<class Allocator> struct hash<basic_stacktrace<Allocator>>;

  The specializations are enabled ([unord.hash]).
*/

#include <cassert>
#include <stacktrace>

int main(int, char**) {
  std::stacktrace empty_trace;
  std::stacktrace nonempty_trace = std::stacktrace::current();

  size_t empty_hash    = std::hash<std::stacktrace>()(empty_trace);
  size_t nonempty_hash = std::hash<std::stacktrace>()(nonempty_trace);
  assert(empty_hash != nonempty_hash);

  std::stacktrace_entry const empty_entry;
  std::stacktrace_entry const& nonempty_entry = nonempty_trace[0];

  size_t empty_entry_hash    = std::hash<std::stacktrace_entry>()(empty_entry);
  size_t nonempty_entry_hash = std::hash<std::stacktrace_entry>()(nonempty_entry);
  assert(empty_entry_hash != nonempty_entry_hash);

  return 0;
}
