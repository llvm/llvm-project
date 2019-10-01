//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

static int foo(int x, int y) {
  return x + y; // BREAK HERE
}

static int bar(int x) {
  return foo(x + 1, x * 2);
}

int main (int argc, char const *argv[])
{
    return bar(argc + 2);
}
