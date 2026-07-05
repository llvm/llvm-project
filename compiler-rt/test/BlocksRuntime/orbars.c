//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 *  orbars.c
 *  testObjects
 *
 *  Created by Blaine Garst on 9/17/08.
 *
 *  CONFIG rdar://6276695 error: before ‘|’ token
 */

#include <stdio.h>

int main(int argc, char *argv[]) {
    int i = 10;
    void (^b)(void) = ^(void){ | i | printf("hello world, %d\n", ++i); };
    printf("%s: success :-(\n", argv[0]);
    return 0;
}
