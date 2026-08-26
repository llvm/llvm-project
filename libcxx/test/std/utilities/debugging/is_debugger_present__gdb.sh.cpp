//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// REQUIRES: host-has-gdb-with-python
// The Android libc++ tests are run on a non-Android host, connected to an
// Android device over adb. gdb needs special support to make this work (e.g.
// gdbclient.py, ndk-gdb.py, gdbserver), and the Android organization doesn't
// support gdb anymore, favoring lldb instead.
// UNSUPPORTED: android
// UNSUPPORTED: asan
// XFAIL: LIBCXX-PICOLIBC-FIXME

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: %{exec} %{gdb} --return-child-result -ex run %t.exe

// <debugging>

// bool is_debugger_present() noexcept;

#include <debugging>

int main(int, char**) {
  static_assert(noexcept(std::is_debugger_present()));
  return std::is_debugger_present() ? 0 : 1;
}
