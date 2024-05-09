//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: libcpp-has-no-incomplete-debugging
// XFAIL: LIBCXX-PICOLIBC-FIXME

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: "%{gdb}" %t.exe -ex "source %S/breakpoint__gdb.cmd" \
// RUN:   | grep -qFf %S/breakpoint__gdb.grep

// <debugging>

// void breakpoint() noexcept;

#include <debugging>

void test() {
  static_assert(noexcept(std::breakpoint()));

  std::breakpoint();
}

int main(int, char**) {
  test();

  return 0;
}
