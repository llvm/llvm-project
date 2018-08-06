//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

extern int a(int);
extern int b(int);
int main (int argc, char const *argv[])
{
    printf ("a(1) returns %d\n", a(1));
    printf ("b(2) returns %d\n", b(2));
}
