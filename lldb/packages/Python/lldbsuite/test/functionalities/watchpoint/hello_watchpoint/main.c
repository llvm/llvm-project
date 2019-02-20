//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <stdint.h>

int32_t global = 10; // Watchpoint variable declaration.
char gchar1 = 'a';
char gchar2 = 'b';

int main(int argc, char** argv) {
    int local = 0;
    printf("&global=%p\n", &global);
    printf("about to write to 'global'...\n"); // Set break point at this line.
                                               // When stopped, watch 'global' for write.
    global = 20;
    gchar1 += 1;
    gchar2 += 1;
    local += argc;
    ++local;
    printf("local: %d\n", local);
    printf("global=%d\n", global);
    printf("gchar1='%c'\n", gchar1);
    printf("gchar2='%c'\n", gchar2);
}
