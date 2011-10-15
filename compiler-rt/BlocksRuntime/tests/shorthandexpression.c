//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

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
