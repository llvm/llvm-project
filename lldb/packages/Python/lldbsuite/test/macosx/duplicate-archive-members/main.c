//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
