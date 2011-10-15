//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

// CONFIG open rdar://6718399
#include <Block.h>

void foo() {
    void (^bbb)(void)  = Block_copy(^ {
        int j, cnt;
    });
}
