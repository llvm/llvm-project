//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

#include <stacktrace>

#include <cassert>

/*
  (19.6.6) Hash Support

  template<> struct hash<stacktrace_entry>;                                 [1]
  template<class Allocator> struct hash<basic_stacktrace<Allocator>>;       [2]

  The specializations are enabled ([unord.hash]).
*/

int main(int, char**) {
  /*
  [1]
  template<> struct hash<stacktrace_entry>;
  */
  std::stacktrace_entry empty_entry;
  assert(std::hash<std::stacktrace_entry>()(empty_entry) == 0);

  /*
  [2]
  template<class Allocator> struct hash<basic_stacktrace<Allocator>>;
  */
  std::stacktrace trace; // initially empty
  auto hash_val_empty    = std::hash<std::stacktrace>()(trace);
  trace                  = /* reassign */ std::stacktrace::current();
  auto hash_val_nonempty = std::hash<std::stacktrace>()(trace);

  assert(hash_val_empty != hash_val_nonempty);

  return 0;
}
