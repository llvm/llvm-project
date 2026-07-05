//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//
//  byrefcopystack.m
//  testObjects
//
//  Created by Blaine Garst on 5/13/08.
//



#include <stdio.h>
#include <Block.h>

// CONFIG rdar://6255170

void (^bumpi)(void);
int (^geti)(void);

void setClosures() {
    int __block i = 10;
    bumpi = Block_copy(^{ ++i; });
    geti = Block_copy(^{ return i; });
}

int main(int argc, char *argv[]) {
    setClosures();
    bumpi();
    int i = geti();
    
    if (i != 11) {
        printf("*** %s didn't update i\n", argv[0]);
        return 1;
    }
    printf("%s: success\n", argv[0]);
    return 0;
}
