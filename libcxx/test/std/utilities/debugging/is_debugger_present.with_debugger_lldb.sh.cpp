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

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: %{lldb} %t.exe -o run -o detach -o quit
// RUN: %{gdb} %t.exe -ex run -ex detach -ex quit --silent

// <debugging>

// bool is_debugger_present() noexcept;

#include <cassert>
#include <debugging>
#include <cstdlib>

// Test with debugger attached:

// LLDB command: `lldb "is_debugger_present.pass" -o run -o detach -o quit`
// GDB command:  `gdb is_debugger_present.pass -ex run -ex detach -ex quit --silent`

void test() {
  static_assert(noexcept(std::is_debugger_present()));

  assert(std::is_debugger_present());
}

int main(int, char**) {
  test();

  return 0;
}
