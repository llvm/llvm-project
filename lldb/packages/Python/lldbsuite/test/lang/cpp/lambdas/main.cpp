//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

int main (int argc, char const *argv[])
{
    printf("Stop here\n"); //% self.runCmd("expression auto $add = [](int first, int second) { return first + second; }")
                           //% self.expect("expression $add(2,3)", substrs = ['= 5'])
    return 0;
}
