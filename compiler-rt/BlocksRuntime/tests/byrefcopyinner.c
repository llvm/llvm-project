//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

#include <Block.h>
#include <stdio.h>

// CONFIG rdar://6225809
// fixed in 5623

int main(int argc, char *argv[]) {
    __block int a = 42;
    int* ap = &a; // just to keep the address on the stack.

    void (^b)(void) = ^{
        //a;              // workaround, a should be implicitly imported
        Block_copy(^{
            a = 2;
        });
    };

    Block_copy(b);

    if(&a == ap) {
        printf("**** __block heap storage should have been created at this point\n");
        return 1;
    }
    printf("%s: Success\n", argv[0]);
    return 0;
}
