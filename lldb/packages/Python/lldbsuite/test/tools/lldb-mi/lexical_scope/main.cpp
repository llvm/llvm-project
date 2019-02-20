//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

void
some_func (void)
{
}

void test_sibling_scope (void)
{
  int a = 1;
  {
    int b = 2;
    some_func(); // BP_first
  }
  {
    short b = 3;
    some_func(); // BP_second
  }
}

int
main (int argc, char **argv)
{
  test_sibling_scope();
  return 0;
}
