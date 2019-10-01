//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

