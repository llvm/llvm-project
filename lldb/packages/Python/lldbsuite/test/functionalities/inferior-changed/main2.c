//===-- main2.c -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char* argv[])
{
    int *int_ptr = (int *)malloc(sizeof(int));
    *int_ptr = 7;
    printf("Hello, world!\n");
    printf("Now not crash %d\n", *int_ptr); // Not crash here.
}
