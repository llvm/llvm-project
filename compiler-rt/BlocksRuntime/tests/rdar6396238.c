//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// CONFIG rdar://6396238

#include <stdio.h>
#include <stdlib.h>

static int count = 0;

void (^mkblock(void))(void)
{
    count++;
    return ^{
        count++;
    };
}

int main (int argc, const char * argv[]) {
    mkblock()();
    if (count != 2) {
        printf("%s: failure, 2 != %d\n", argv[0], count);
        exit(1);
    } else {
        printf("%s: success\n", argv[0]);
        exit(0);
    }
    return 0;
}
