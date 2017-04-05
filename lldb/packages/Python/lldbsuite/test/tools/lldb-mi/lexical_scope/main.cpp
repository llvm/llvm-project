//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
