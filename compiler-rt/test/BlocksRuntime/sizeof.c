//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
