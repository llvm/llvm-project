//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 *  copynull.c
 *  testObjects
 *
 *  Created by Blaine Garst on 10/15/08.
 *
 */
 
#import <stdio.h>
#import <Block.h>
#import <Block_private.h>
 
// CONFIG rdar://6295848

int main(int argc, char *argv[]) {
    
    void (^block)(void) = (void (^)(void))0;
    void (^blockcopy)(void) = Block_copy(block);
    
    if (blockcopy != (void (^)(void))0) {
        printf("whoops, somehow we copied NULL!\n");
        return 1;
    }
    // make sure we can also
    Block_release(blockcopy);
    // and more secretly
    //_Block_destroy(blockcopy);
    
    printf("%s: success\n", argv[0]);
    return 0;
}
