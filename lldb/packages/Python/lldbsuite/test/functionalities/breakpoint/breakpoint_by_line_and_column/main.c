//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

int square(int x)
{
  return x * x;
}

int main (int argc, char const *argv[])
{
  int did_call = 0;

  // Line 20.                                    v Column 50.
  if(square(argc+1) != 0) { did_call = 1; return square(argc); }
  //                                             ^
  return square(0);
}
