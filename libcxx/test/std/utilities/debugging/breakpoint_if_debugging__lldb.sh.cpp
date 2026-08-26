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

// RUN: %{cxx} %{flags} %s %{compile_flags} %{link_flags} -o %t.exe -g
// RUN: %if darwin %{ codesign --entitlements %S/entitlements-macos.plist -f -s - %t.exe %}
// RUN: %{exec} "%{lldb}" %t.exe -o "command script import %S/breakpoint__lldb.py"
// RUN: %{exec} %t.exe

// <debugging>

// void breakpoint_if_debugging() noexcept;

#include <debugging>

int main(int, char**) {
  static_assert(noexcept(std::breakpoint_if_debugging()));
  std::breakpoint_if_debugging();

  return 0;
}
