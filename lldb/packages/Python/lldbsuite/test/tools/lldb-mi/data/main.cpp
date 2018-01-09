//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

const char g_CharArray[] = "\x10\x11\x12\x13";
static const char s_CharArray[] = "\x20\x21\x22\x23";

void
local_array_test_inner()
{
    char array[] = { 0x01, 0x02, 0x03, 0x04 };
    char *first_element_ptr = &array[0];
    // BP_local_array_test_inner
    return;
}

void
local_array_test()
{
    char array[] = { 0x05, 0x06, 0x07, 0x08 };
    // BP_local_array_test
    local_array_test_inner();
    return;
}

void
local_2d_array_test()
{
    int array2d[2][3];
    array2d[0][0] = 1;
    array2d[0][1] = 2;
    array2d[0][2] = 3;
    array2d[1][0] = 4;
    array2d[1][1] = 5;
    array2d[1][2] = 6;
    return; // BP_local_2d_array_test
}

void
hello_world()
{
    printf("Hello, World!\n"); // BP_hello_world
}

int
main(int argc, char const *argv[])
{ // FUNC_main
    local_array_test();
    hello_world();
    local_2d_array_test();
    return 0;
}
