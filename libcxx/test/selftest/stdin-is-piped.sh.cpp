//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that the executor pipes standard input to the test-executable being run.

// RUN: %{build}
// RUN: echo "abc" | %{exec} %t.exe

#include <cstdio>

int main(int, char**) {
  int input[] = {std::getchar(), std::getchar(), std::getchar()};

  if (input[0] == 'a' && input[1] == 'b' && input[2] == 'c')
    return 0;
  return 1;
}
