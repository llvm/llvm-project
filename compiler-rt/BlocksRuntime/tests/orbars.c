//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

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
