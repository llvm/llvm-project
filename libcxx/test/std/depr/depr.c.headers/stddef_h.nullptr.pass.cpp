//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <stddef.h>

#include <stddef.h>
#include <cassert>

int main(int, char**) {
    {
        void *p = NULL;
        assert(!p);
    }
    {
        void *p = nullptr;
        assert(!p);
    }

    return 0;
}
