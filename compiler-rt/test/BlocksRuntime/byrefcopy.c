//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//
//  byrefcopy.m
//  testObjects
//
//  Created by Blaine Garst on 5/13/08.
//

#include <stdio.h>
#include <Block.h>
#include <Block_private.h>

// CONFIG

void callVoidVoid(void (^closure)(void)) {
    closure();
}

int main(int argc, char *argv[]) {
    int __block i = 10;
    
    void (^block)(void) = ^{ ++i; };
    //printf("original (old style) is  %s\n", _Block_dump_old(block));
    //printf("original (new style) is %s\n", _Block_dump(block));
    void (^blockcopy)(void) = Block_copy(block);
    //printf("copy is %s\n", _Block_dump(blockcopy));
    // use a copy & see that it updates i
    callVoidVoid(block);
    
    if (i != 11) {
        printf("*** %s didn't update i\n", argv[0]);
        return 1;
    }
    printf("%s: success\n", argv[0]);
    return 0;
}
