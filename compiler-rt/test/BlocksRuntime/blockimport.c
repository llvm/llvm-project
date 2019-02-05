//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 *  blockimport.c
 *  testObjects
 *
 *  Created by Blaine Garst on 10/13/08.
 *
 */


//
// pure C nothing more needed
// CONFIG  rdar://6289344

#include <stdio.h>
#include <Block.h>
#include <Block_private.h>




int main(int argc, char *argv[]) {
    int i = 1;
    int (^intblock)(void) = ^{ return i*10; };
    
    void (^vv)(void) = ^{
        if (argc > 0) {
            printf("intblock returns %d\n", intblock());
        }
    };

#if 0    
    //printf("Block dump %s\n", _Block_dump(vv));
    {
        struct Block_layout *layout = (struct Block_layout *)(void *)vv;
        printf("isa %p\n", layout->isa);
        printf("flags %x\n", layout->flags);
        printf("descriptor %p\n", layout->descriptor);
        printf("descriptor->size %d\n", layout->descriptor->size);
    }
#endif
    void (^vvcopy)(void) = Block_copy(vv);
    Block_release(vvcopy);
    printf("%s: success\n", argv[0]);
    return 0;
}
