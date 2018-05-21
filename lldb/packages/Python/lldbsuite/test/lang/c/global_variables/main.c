//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

int g_common_1; // Not initialized on purpose to cause it to be undefined external in .o file
int g_file_global_int = 42;
static const int g_file_static_int = 2;
const char *g_file_global_cstr = "g_file_global_cstr";
static const char *g_file_static_cstr = "g_file_static_cstr";
int *g_ptr = &g_file_global_int;

extern int g_a;
int main (int argc, char const *argv[])
{
    g_common_1 = g_file_global_int / g_file_static_int;
    static const char *g_func_static_cstr = "g_func_static_cstr";
    printf ("%s %s\n", g_file_global_cstr, g_file_static_cstr);
    return g_file_global_int + g_a + g_common_1 + *g_ptr; // Set break point at this line.  //// break $source:$line; continue; var -global g_a -global g_global_int
}
