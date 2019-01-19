//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <unistd.h>

int do_loop;
int do_segfault;

int
main(int argc, char const *argv[])
{
    if (do_loop)
    {
        do
            sleep(1);
        while (do_loop); // BP_loop_condition
    }

    if (do_segfault)
    {
        int *null_ptr = NULL;
        return *null_ptr;
    }

    return 0;
}
