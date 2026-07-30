//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: LIBCXX-PICOLIBC-FIXME

// <iostream>

// ostream cerr;

// Unlike cin/cout/clog, std::cerr is still unbuffered even when sync_with_stdio(false)
// is called. This test ensures that it keeps producing correctly-ordered output when
// interleaved with the C stdio functions even after sync_with_stdio(false).

// RUN: %{build}
// RUN: %{exec} %t.exe 2> %t.actual
// RUN: echo -n 1234 > %t.expected
// RUN: diff %t.expected %t.actual

#include <cassert>
#include <cstdio>
#include <iostream>

int main(int, char**) {
  assert(std::ios_base::sync_with_stdio(false));

  // interleave output with C stdio functions
  std::fputs("1", stderr);
  std::cerr << "2";
  std::fprintf(stderr, "%d", 3);
  std::cerr << 4;

  return 0;
}
