//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// XFAIL: LIBCXX-AIX-FIXME, LIBCXX-PICOLIBC-FIXME
// UNSUPPORTED: gcc
// UNSUPPORTED: android

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: %{lldb} %t.exe -o run -o detach -o quit
// RUN: %{gdb} %t.exe -ex run -ex detach -ex quit --silent

// <debugging>

// void breakpoint() noexcept;

#include <cassert>
#include <debugging>

// Test with debugger attached:

// LLDB command: `lldb "breakpoint.pass" -o run -o detach -o quit`
// GDB command:  `gdb breakpoint.pass -ex run -ex detach -ex quit --silent`

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

// https://lldb.llvm.org/use/python-reference.html#running-a-python-script-when-a-breakpoint-gets-hit

void test() {
  static_assert(noexcept(std::breakpoint()));

  std::breakpoint();
}

int main(int, char**) {
  test();

  return 0;
}
