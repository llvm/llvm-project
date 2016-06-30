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
uint8_t byteArray[8] = {0};
uint64_t pad1 = 0;
uint16_t wordArray[4] = {0};
uint64_t pad2 = 0;
uint32_t dwordArray[2] = {0};

int main(int argc, char** argv) {

    int i;
    uint8_t localByte;
    uint16_t localWord;
    uint32_t localDword;

    for (i = 0; i < 8; i++)
    {
        printf("About to write byteArray[%d] ...\n", i); // About to write byteArray
        pad0++;
        byteArray[i] = 7;
        pad1++;
        localByte = byteArray[i]; // Here onwards we should'nt be stopped in loop
        byteArray[i]++;
        localByte = byteArray[i];
    }

    pad0 = 0;
    pad1 = 0;

    for (i = 0; i < 4; i++)
    {
        printf("About to write wordArray[%d] ...\n", i); // About to write wordArray
        pad0++;
        wordArray[i] = 7;
        pad1++;
        localWord = wordArray[i]; // Here onwards we should'nt be stopped in loop
        wordArray[i]++;
        localWord = wordArray[i];
    }

    pad0 = 0;
    pad1 = 0;

    for (i = 0; i < 2; i++)
    {
        printf("About to write dwordArray[%d] ...\n", i); // About to write dwordArray
        pad0++;
        dwordArray[i] = 7;
        pad1++;
        localDword = dwordArray[i]; // Here onwards we shouldn't be stopped in loop
        dwordArray[i]++;
        localDword = dwordArray[i];
    }

    return 0;
}
