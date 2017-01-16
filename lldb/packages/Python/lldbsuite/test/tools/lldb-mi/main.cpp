//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
