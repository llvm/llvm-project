//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

int main (int argc, char const *argv[])
{
  void* my_ptr[] = {
    reinterpret_cast<void*>(0xDEADBEEF),
    reinterpret_cast<void*>(main),
    reinterpret_cast<void*>(0xFEEDBEEF),
    reinterpret_cast<void*>(0xFEEDDEAD),
    reinterpret_cast<void*>(0xDEADFEED)
  };
  return 0; // break here
}

