//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdio>

void
g_MyFunction(void)
{
    printf("g_MyFunction");
}

static void
s_MyFunction(void)
{
    g_MyFunction();
    printf("s_MyFunction");
}

int
main(int argc, char const *argv[])
{
    printf("start");
    g_MyFunction();
    s_MyFunction();
    printf("exit");
    return 0;
}
