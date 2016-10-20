//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdint.h>

uint64_t pad0 = 0;
uint8_t byteArray[4] = {0};
uint64_t pad1 = 0;

int main(int argc, char** argv) {

    int i;

    for (i = 0; i < 4; i++)
    {
        printf("About to write byteArray[%d] ...\n", i); // About to write byteArray
        pad0++;
        byteArray[i] = 7;
        pad1++;
    }

    return 0;
}
