//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23
// UNSUPPORTED: libcpp-has-no-incomplete-debugging
// REQUIRES: host-has-lldb-with-python
// The Android libc++ tests are run on a non-Android host, connected to an
// Android device over adb.
// UNSUPPORTED: android
// XFAIL: LIBCXX-PICOLIBC-FIXME

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: "%{lldb}" %t.exe -o "command source %S/breakpoint_if_debugging__lldb.cmd" \
// RUN:   | grep -qEf %S/breakpoint_if_debugging__lldb.grep

// <debugging>

// void breakpoint_if_debugging() noexcept;

#include <debugging>

void test() {
  static_assert(noexcept(std::breakpoint_if_debugging()));

  std::breakpoint_if_debugging();
}

int main(int, char**) {
  test();

  return 0;
}
