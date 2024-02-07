//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// REQUIRES: host-has-gdb-with-python
// UNSUPPORTED: android
// XFAIL: LIBCXX-AIX-FIXME, LIBCXX-PICOLIBC-FIXME

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: %{gdb} %t.exe -ex "source %S/breakpoint_if_debugging_with_debugger_gdb.py" --silent

// <debugging>

// void breakpoint_if_debugging() noexcept;

#include <cassert>
#include <debugging>

#include "test_macros.h"

#ifdef TEST_COMPILER_GCC
#  define OPT_NONE __attribute__((noinline))
#else
#  define OPT_NONE __attribute__((optnone))
#endif

void StopForDebugger() OPT_NONE;
void StopForDebugger() {}

// Test with debugger attached:

// GDB command:  `gdb breakpoint_if_debugging.pass -ex run -ex detach -ex quit --silent`

void test() {
  static_assert(noexcept(std::breakpoint_if_debugging()));

  std::breakpoint_if_debugging();
}

int main(int, char**) {
  test();

  return 0;
}
