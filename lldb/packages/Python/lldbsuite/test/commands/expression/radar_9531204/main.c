//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

// breakpoint set -n foo
//
//
int foo (int value)
{
  printf ("I got the value: %d.\n", value);
  return 0;
}

int main (int argc, char **argv)
{
  foo (argc);
  printf ("Hello there: %d.\n", argc);
  return 0;
}
