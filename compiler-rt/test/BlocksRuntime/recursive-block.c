//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <Block.h>
#include <Block_private.h>
#include <stdlib.h>

// CONFIG


int cumulation = 0;

int doSomething(int i) {
    cumulation += i;
    return cumulation;
}

void dirtyStack() {
    int i = random();
    int j = doSomething(i);
    int k = doSomething(j);
    doSomething(i + j + k);
}

typedef void (^voidVoid)(void);

voidVoid testFunction() {
    int i = random();
    __block voidVoid inner = ^{ doSomething(i); };
    //printf("inner, on stack, is %p\n", (void*)inner);
    /*__block*/ voidVoid outer = ^{
        //printf("will call inner block %p\n", (void *)inner);
        inner();
    };
    //printf("outer looks like: %s\n", _Block_dump(outer));
    voidVoid result = Block_copy(outer);
    //Block_release(inner);
    return result;
}


int main(int argc, char **argv) {
    voidVoid block = testFunction();
    dirtyStack();
    block();
    Block_release(block);

    printf("%s: success\n", argv[0]);

    return 0;
}
