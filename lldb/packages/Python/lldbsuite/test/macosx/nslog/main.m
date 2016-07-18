//===-- main.m --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <Foundation/Foundation.h>

int main(int argc, char** argv)
{
    printf("About to log\n"); // break here
    NSLog(@"This is a message from NSLog");

    return 0;
}
