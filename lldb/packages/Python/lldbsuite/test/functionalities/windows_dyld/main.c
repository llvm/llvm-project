//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <windows.h>

int __declspec(dllimport) DllFunc(int n);

int main(int argc, char ** argv) {
  int x = DllFunc(4);
  int y = DllFunc(8);   // set breakpoint here
  int z = DllFunc(16);
  return x + y + z;
}
