//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -O0 -g

#include <cassert>
#include <iostream>
#include <stacktrace>

int main(int, char**) {
  // uint32_t line_number = __LINE__ + 1; // record where `current` is being called:
  auto trace = std::stacktrace::current();
  std::cout << trace << '\n';
  // First entry of this should be `main`.
  auto entry = trace.at(0);
  assert(entry);
  assert(entry.native_handle());
  // assert(entry.description() == "main" || entry.description() == "_main");
  // assert(entry.source_file().ends_with(".pass.cpp")); // this cpp file, and not t.tmp.exe
  // assert(entry.source_line() == line_number);
  return 0;
}
