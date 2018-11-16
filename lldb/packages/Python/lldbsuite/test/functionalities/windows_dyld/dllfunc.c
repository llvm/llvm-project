//===-- a.c -----------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <windows.h>

BOOL WINAPI DllMain(HINSTANCE h, DWORD reason, void* reserved) {
  return TRUE;
}

int __declspec(dllexport) DllFunc(int n) {
  int x = n * n;  
  return x;  // set breakpoint here
}
