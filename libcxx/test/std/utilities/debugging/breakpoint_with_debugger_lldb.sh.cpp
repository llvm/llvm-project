//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// REQUIRES: host-has-lldb-with-python
// UNSUPPORTED: android
// XFAIL: LIBCXX-AIX-FIXME, LIBCXX-PICOLIBC-FIXME

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: %{lldb} %t.exe -o "command script import %S/breakpoint_with_debugger_lldb.py" -o run -o detach -o quit

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

// LLDB command: `lldb "breakpoint.pass" -o run -o detach -o quit`

//
// Sample LLDB output:
//
// Process 43162 launched: '/home/llvm-dev/Projects/llvm-project/build/breakpoint.pass' (x86_64)
// Process 43162 stopped
// * thread #1, name = 'breakpoint.pass', stop reason = signal SIGTRAP
//     frame #0: 0x00007ffff7eb27e5 libc++.so.1`std::__1::__breakpoint() at debugging.cpp:44:1
//    41  	#  else
//    42  	  raise(SIGTRAP);
//    43  	#  endif
// -> 44  	}
//    45
//    46  	#else
//    47
// (lldb) detach
// Process 43162 detached
// (lldb) quit

void test() {
  static_assert(noexcept(std::breakpoint()));

  std::breakpoint();
}

int main(int, char**) {
  test();

  return 0;
}
