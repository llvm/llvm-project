//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// ostream cout;

// In synchronized mode, output written through std::cout and through the C stdio
// functions must appear in the order in which it was written.

// RUN: %{build}
// RUN: %{exec} %t.exe > %t.actual
// RUN: echo -n 123456 > %t.expected
// RUN: diff %t.expected %t.actual

#include <cassert>
#include <cstdio>
#include <iostream>

int main(int, char**) {
  std::fputs("1", stdout);
  std::cout << "2";
  std::fprintf(stdout, "%d", 3);
  std::cout << 4;
  std::fputs("5", stdout);
  std::cout << 6;

  return 0;
}
