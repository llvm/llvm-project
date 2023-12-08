//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 *  objectassign.c
 *  testObjects
 *
 *  Created by Blaine Garst on 10/28/08.
 *
 * This just tests that the compiler is issuing the proper helper routines
 * CONFIG C rdar://6175959
 */


#include <stdio.h>
#include <Block_private.h>


int AssignCalled = 0;
int DisposeCalled = 0;

// local copy instead of libSystem.B.dylib copy
void _Block_object_assign(void *destAddr, const void *object, const int isWeak) {
    //printf("_Block_object_assign(%p, %p, %d) called\n", destAddr, object, isWeak);
    AssignCalled = 1;
}

void _Block_object_dispose(const void *object, const int isWeak) {
    //printf("_Block_object_dispose(%p, %d) called\n", object, isWeak);
    DisposeCalled = 1;
}

struct MyStruct {
    long isa;
    long field;
};

typedef struct MyStruct *__attribute__((NSObject)) MyStruct_t;

int main(int argc, char *argv[]) {
    if (__APPLE_CC__ < 5627) {
        printf("need compiler version %d, have %d\n", 5627, __APPLE_CC__);
        return 0;
    }
    // create a block
    struct MyStruct X;
    MyStruct_t xp = (MyStruct_t)&X;
    xp->field = 10;
    void (^myBlock)(void) = ^{ printf("field is %ld\n", xp->field); };
    // should be a copy helper generated with a calls to above routines
    // Lets find out!
    struct Block_layout *bl = (struct Block_layout *)(void *)myBlock;
    if ((bl->flags & BLOCK_HAS_COPY_DISPOSE) != BLOCK_HAS_COPY_DISPOSE) {
        printf("no copy dispose!!!!\n");
        return 1;
    }
    // call helper routines directly.  These will, in turn, we hope, call the stubs above
    long destBuffer[256];
    //printf("destbuffer is at %p, block at %p\n", destBuffer, (void *)bl);
    //printf("dump is %s\n", _Block_dump(myBlock));
    bl->descriptor->copy(destBuffer, bl);
    bl->descriptor->dispose(bl);
    if (AssignCalled == 0) {
        printf("did not call assign helper!\n");
        return 1;
    }
    if (DisposeCalled == 0) {
        printf("did not call dispose helper\n");
        return 1;
    }
    printf("%s: Success!\n", argv[0]);
    return 0;
}
