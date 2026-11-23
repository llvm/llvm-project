//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-PICOLIBC-FIXME

// UNSUPPORTED: no-wide-characters

// <iostream>

// ostream wclog;

// Test that we output correctly after setting sync_with_stdio(false).
// This must be its own test because sync_with_stdio() must be called
// before any other IO operation for the test to be portable.

// RUN: %{build}
// RUN: %{exec} %t.exe 2> %t.actual
// RUN: echo -n 1234 > %t.expected
// RUN: diff %t.expected %t.actual

#include <cassert>
#include <iostream>

int main(int, char**) {
  // flip-flop the setting to check that construction both ways works as expected
  assert(std::ios::sync_with_stdio(false));
  assert(!std::ios::sync_with_stdio(true));
  assert(std::ios::sync_with_stdio(false));
  std::wclog << L"1234";
  return 0;
}
