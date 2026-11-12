//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CONFIG open rdar://6718399
#include <Block.h>

void foo() {
    void (^bbb)(void)  = Block_copy(^ {
        int j, cnt;
    });
}
