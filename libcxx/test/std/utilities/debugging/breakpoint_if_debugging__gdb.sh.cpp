//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// REQUIRES: host-has-gdb-with-python
// UNSUPPORTED: android
// UNSUPPORTED: availability-debugging-missing
// LeakSanitizer does not work under ptrace
// UNSUPPORTED: asan
// XFAIL: LIBCXX-PICOLIBC-FIXME

// RUN: %{cxx} %{flags} %s %{compile_flags} %{link_flags} -o %t.exe -g
// RUN: %{exec} "%{gdb}" %t.exe -ex "source %S/breakpoint__gdb.py"
// RUN: %{exec} %t.exe

// breakpoint_if_debugging() noexcept
#include <debugging>

int main(int, char**) {
  std::breakpoint_if_debugging();

  return 0;
}
