//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

/*
 *  byrefcopyint.c
 *  testObjects
 *
 *  Created by Blaine Garst on 12/1/08.
 *
 */

//
//  byrefcopyid.m
//  testObjects
//
//  Created by Blaine Garst on 5/13/08.
//

// Tests copying of blocks with byref ints
// CONFIG rdar://6414583 -C99

#include <stdio.h>
#include <string.h>
#include <Block.h>
#include <Block_private.h>




typedef void (^voidVoid)(void);

voidVoid dummy;

void callVoidVoid(voidVoid closure) {
    closure();
}


voidVoid testRoutine(const char *whoami) {
    __block int  dumbo = strlen(whoami);
    dummy = ^{
        //printf("incring dumbo from %d\n", dumbo);
        ++dumbo;
    };
    
    
    voidVoid copy = Block_copy(dummy);
    

    return copy;
}

int main(int argc, char *argv[]) {
    voidVoid array[100];
    for (int i = 0; i <  100; ++i) {
        array[i] = testRoutine(argv[0]);
        array[i]();
    }
    for (int i = 0; i <  100; ++i) {
        Block_release(array[i]);
    }
    
    
    printf("%s: success\n", argv[0]);
    return 0;
}
