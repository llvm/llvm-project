//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

/*
 *  voidarg.c
 *  testObjects
 *
 *  Created by Blaine Garst on 2/17/09.
 *
 */

// PURPOSE should complain about missing 'void' but both GCC and clang are supporting K&R instead
// CONFIG open error:

#include <stdio.h>

int Global;

void (^globalBlock)() = ^{ ++Global; };         // should be void (^gb)(void) = ...

int main(int argc, char *argv[]) {
    printf("%s: success", argv[0]);
    return 0;
}
