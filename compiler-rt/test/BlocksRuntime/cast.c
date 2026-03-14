//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
