//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>


int f1 (char *s);
int f2 (char *s);
int f3 (char *s);


// We want f1 to start on line 20
int f1 (char *s)
{
    return printf("f1: %s\n", s);
}





// We want f2 to start on line 30, this should get stripped
int f2 (char *s)
{
    return printf("f2: %s\n", s);
}





// We want f3 to start on line 40
int f3 (char *s)
{
    return printf("f3: %s\n", s);
}





// We want main to start on line 50
int main (int argc, const char * argv[])
{
    f1("carp");
    f3("dong");
    return 0;
}
