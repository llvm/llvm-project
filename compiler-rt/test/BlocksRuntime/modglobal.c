//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <stdio.h>

// CONFIG

int AGlobal;

int main(int argc, char *argv[]) {
    void (^f)(void) = ^ { AGlobal++; };
    
    printf("%s: success\n", argv[0]);
    return 0;

}
