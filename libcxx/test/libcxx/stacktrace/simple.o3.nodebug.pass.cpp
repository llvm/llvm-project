//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -O3 -g0

#include <cassert>
#include <stacktrace>
#include <iostream>

int main(int, char**) {
  // Get the current trace.
  // uint32_t line_number = __LINE__ + 1; // record where `current` is being called:
  auto trace = std::stacktrace::current();
  std::cout << trace << std::endl;
  // First entry of this should be `main`.
  auto entry = trace.at(0);
  assert(entry);
  assert(entry.native_handle());
  assert(entry.description() == "main" || entry.description() == "_main");
  // This is 'nodebug', so we cannot get the source file and line:
  // assert(entry.source_file().ends_with(".pass.cpp"));
  // assert(entry.source_line() == line_number);
  // But this should at least produce the executable filename
  assert(entry.source_file().contains("simple.o3.nodebug.pass.cpp"));
  return 0;
}
