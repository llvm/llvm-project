//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdio>

int
main(int argc, char const *argv[])
{
    printf("argc=%d\n", argc);  // BP_printf
    for (int i = 0; i < argc; ++i)
        printf("argv[%d]=%s\n", i, argv[i]);
    return 0;   // BP_return
}
