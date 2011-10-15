//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
#include <stdio.h>

// CONFIG

int AGlobal;

int main(int argc, char *argv[]) {
    void (^f)(void) = ^ { AGlobal++; };
    
    printf("%s: success\n", argv[0]);
    return 0;

}
