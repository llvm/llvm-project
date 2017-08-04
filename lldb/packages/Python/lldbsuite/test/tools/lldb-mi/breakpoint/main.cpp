//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstdio>

namespace ns
{
    int foo1(void) { printf("In foo1\n"); return 1; }
    int foo2(void) { printf("In foo2\n"); return 2; }
}

// BP_before_main

int x;
int
main(int argc, char const *argv[])
{
    printf("Print a formatted string so that GCC does not optimize this printf call: %s\n", argv[0]);
    x = ns::foo1() + ns::foo2();
    return 0; // BP_return
}
