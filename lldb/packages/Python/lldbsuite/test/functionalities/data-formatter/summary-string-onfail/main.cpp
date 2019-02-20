//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

struct contained
{
    int first;
    int second;
};

struct container
{
    int scalar;
    struct contained *pointer;
};

int main ()
{
    struct container mine = {1, 0};
    printf ("Mine's scalar is the only thing that is good: %d.\n", mine.scalar); // Set break point at this line.
    return 0;
}

