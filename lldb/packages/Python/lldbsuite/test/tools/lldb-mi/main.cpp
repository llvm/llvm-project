//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdio>

#ifdef _WIN32
    #include <direct.h>
    #define getcwd _getcwd // suppress "deprecation" warning
#else
    #include <unistd.h>
#endif

int
main(int argc, char const *argv[])
{
    int a = 10;

    char buf[512];
    char *ans = getcwd(buf, sizeof(buf));
    if (ans) {
        printf("cwd: %s\n", ans);
    }

    printf("argc=%d\n", argc); // BP_printf

    return 0;
}
