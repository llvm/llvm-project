//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdio>

namespace ns
{
    int foo1(void) { printf("In foo1\n"); return 1; }
    int foo2(void) { printf("In foo2\n"); return 2; }
}

int x;
int main(int argc, char const *argv[]) { // BP_main_decl
    printf("Print a formatted string so that GCC does not optimize this printf call: %s\n", argv[0]);
  // This is a long comment with no code inside
  // This is a long comment with no code inside
  // This is a long comment with no code inside
  // BP_in_main
  // This is a long comment with no code inside
  // This is a long comment with no code inside
  // This is a long comment with no code inside
    x = ns::foo1() + ns::foo2();
    return 0; // BP_return
}
