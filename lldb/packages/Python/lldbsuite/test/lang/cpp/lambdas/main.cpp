//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

int main (int argc, char const *argv[])
{
    printf("Stop here\n"); //% self.runCmd("expression auto $add = [](int first, int second) { return first + second; }")
                           //% self.expect("expression $add(2,3)", substrs = ['= 5'])
    return 0;
}
