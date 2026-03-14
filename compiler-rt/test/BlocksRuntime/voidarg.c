//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
