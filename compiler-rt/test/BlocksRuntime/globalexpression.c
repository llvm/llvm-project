//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// testfilerunner CONFIG

#import <stdio.h>
#import <Block.h>

int global;

void (^gblock)(int) = ^(int x){ global = x; };

int main(int argc, char *argv[]) {
    gblock(1);
    if (global != 1) {
        printf("%s: *** did not set global to 1\n", argv[0]);
        return 1;
    }
    void (^gblockcopy)(int) = Block_copy(gblock);
    if (gblockcopy != gblock) {
        printf("global copy %p not a no-op %p\n", (void *)gblockcopy, (void *)gblock);
        return 1;
    }
    Block_release(gblockcopy);
    gblock(3);
    if (global != 3) {
        printf("%s: *** did not set global to 3\n", argv[0]);
        return 1;
    }
    gblockcopy = Block_copy(gblock);
    gblockcopy(5);
    if (global != 5) {
        printf("%s: *** did not set global to 5\n", argv[0]);
        return 1;
    }
    printf("%s: Success!\n", argv[0]);
    return 0;
}

