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

// wostream wcerr;

// Unlike wcin/wcout/wclog, std::wcerr is still unbuffered even when sync_with_stdio(false)
// is called. This test ensures that it produces correct output after sync_with_stdio(false).
// We can't check interleaving with C stdio functions here since C stdio uses narrow characters.

// RUN: %{build}
// RUN: %{exec} %t.exe 2> %t.actual
// RUN: echo -n 1234 > %t.expected
// RUN: diff %t.expected %t.actual

#include <cassert>
#include <iostream>

int main(int, char**) {
  assert(std::ios_base::sync_with_stdio(false));

  std::wcerr << L"12" << 34;

  return 0;
}
