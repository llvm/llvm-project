//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

/*
 *  sizeof.c
 *  testObjects
 *
 *  Created by Blaine Garst on 2/17/09.
 *
 */

#include <stdio.h>

// CONFIG error:

int main(int argc, char *argv[]) {

    void (^aBlock)(void) = ^{ printf("hellow world\n"); };

    printf("the size of a block is %ld\n", sizeof(*aBlock));
    printf("%s: success\n", argv[0]);
    return 0;
}
