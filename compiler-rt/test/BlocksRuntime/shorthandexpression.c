//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 *  shorthandexpression.c
 *  testObjects
 *
 *  Created by Blaine Garst on 9/16/08.
 *
 *  CONFIG error:
 */


void foo() {
    void (^b)(void) = ^(void)printf("hello world\n");
}

int main(int argc, char *argv[]) {
    printf("%s: this shouldn't compile\n", argv[0]);
    return 1;
}
