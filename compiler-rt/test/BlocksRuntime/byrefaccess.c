//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//
//  byrefaccess.m
//  test that byref access to locals is accurate
//  testObjects
//
//  Created by Blaine Garst on 5/13/08.
//
// CONFIG

#include <stdio.h>


void callVoidVoid(void (^closure)(void)) {
    closure();
}

int main(int argc, char *argv[]) {
    __block int i = 10;
    
    callVoidVoid(^{ ++i; });
    
    if (i != 11) {
        printf("*** %s didn't update i\n", argv[0]);
        return 1;
    }
    printf("%s: success\n", argv[0]);
    return 0;
}
