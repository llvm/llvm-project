//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// QEMU does not detect EOF, when reading from stdin
// "echo -n" suppresses any characters after the output and so the test hangs.
// https://gitlab.com/qemu-project/qemu/-/issues/1963
// UNSUPPORTED: LIBCXX-PICOLIBC-FIXME

// This test hangs on Android devices that lack shell_v2, which was added in
// Android N (API 24).
// UNSUPPORTED: LIBCXX-ANDROID-FIXME && android-device-api={{2[1-3]}}

// UNSUPPORTED: no-wide-characters

// <iostream>

// istream wcin;

// Test that we output correctly after setting sync_with_stdio(false).
// This must be its own test because sync_with_stdio() must be called
// before any other IO operation for the test to be portable.

// RUN: %{build}
// RUN: echo -n 1234 > %t.input
// RUN: %{exec} %t.exe < %t.input

#include <cassert>
#include <iostream>

int main(int, char**) {
  // flip-flop the setting to check that construction both ways works as expected
  assert(std::ios::sync_with_stdio(false));
  assert(!std::ios::sync_with_stdio(true));
  assert(std::ios::sync_with_stdio(false));
  int i;
  std::wcin >> i;
  assert(i == 1234);
  return 0;
}
