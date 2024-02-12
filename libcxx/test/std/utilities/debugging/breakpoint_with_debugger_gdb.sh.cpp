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
// RUN: %{gdb} %t.exe -ex "source %S/breakpoint_with_debugger_gdb.py" --silent

// <debugging>

// void breakpoint() noexcept;

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

// GDB command:  `gdb breakpoint.pass -ex run -ex detach -ex quit --silent`

// gdb breakpoint.pass -ex run -ex "signal SIGTRAP" -ex detach -ex quit --silent

//
// Sample GDB ouput:
//
// Reading symbols from breakpoint.pass..
// Starting program: /home/llvm-dev/Projects/llvm-project/build/breakpoint.pass
// [Thread debugging using libthread_db enabled]
// Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".

// Program received signal SIGTRAP, Trace/breakpoint trap.
// std::__1::__breakpoint () at /home/llvm-dev/Projects/llvm-project/libcxx/src/debugging.cpp:44
// warning: Source file is more recent than executable.
// 44	}
// Detaching from program: /home/llvm-dev/Projects/llvm-project/build/breakpoint.pass, process 53887
// [Inferior 1 (process 53887) detached]

#include <print>

void test() {
  static_assert(noexcept(std::breakpoint()));

  std::println("1111111");
  // StopForDebugger();
  std::breakpoint();
  StopForDebugger();
  std::println("222222");
}

int main(int, char**) {
  test();

  return 0;
}
