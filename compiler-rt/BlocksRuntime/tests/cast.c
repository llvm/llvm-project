//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

/*
 *  cast.c
 *  testObjects
 *
 *  Created by Blaine Garst on 2/17/09.
 *
 */

// PURPOSE should allow casting of a Block reference to an arbitrary pointer and back
// CONFIG open

#include <stdio.h>



int main(int argc, char *argv[]) {

    void (^aBlock)(void);
    int *ip;
    char *cp;
    double *dp;

    ip = (int *)aBlock;
    cp = (char *)aBlock;
    dp = (double *)aBlock;
    aBlock = (void (^)(void))ip;
    aBlock = (void (^)(void))cp;
    aBlock = (void (^)(void))dp;
    printf("%s: success", argv[0]);
    return 0;
}
