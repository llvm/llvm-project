//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
typedef signed char v4i8 __attribute__ ((vector_size(4)));
v4i8 global_vector = {1, 2, 3, 4};

int
main ()
{
  return 0;
}
