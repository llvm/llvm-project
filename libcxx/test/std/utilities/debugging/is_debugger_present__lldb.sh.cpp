//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// REQUIRES: host-has-lldb
// The Android libc++ tests are run on a non-Android host, connected to an
// Android device over adb.
// UNSUPPORTED: android
// UNSUPPORTED: availability-debugging-missing
// XFAIL: LIBCXX-PICOLIBC-FIXME

// RUN: %{cxx} %{flags} %s -o %t.exe %{compile_flags} -g %{link_flags}
// RUN: %if darwin %{ codesign --entitlements %S/entitlements-macos.plist -f -s - %t.exe %}
// RUN: %{exec} "%{lldb}" %t.exe -o "command script import %S/is_debugger_present__lldb.py"

// <debugging>

// bool is_debugger_present() noexcept;

#include <debugging>

int main(int, char**) {
  static_assert(noexcept(std::is_debugger_present()));
  return std::is_debugger_present() ? 0 : 1;
}
