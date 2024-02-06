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

// void breakpoint_if_debugging() noexcept;

#include <cassert>
#include <debugging>

// Test with debugger attached:

// LLDB command: `lldb "breakpoint_if_debugging.pass" -o run -o detach -o quit`
// GDB command:  `gdb breakpoint_if_debugging.pass -ex run -ex detach -ex quit --silent`

void test() {
  static_assert(noexcept(std::breakpoint_if_debugging()));

  std::breakpoint_if_debugging();
}

int main(int, char**) {
  test();

  return 0;
}
