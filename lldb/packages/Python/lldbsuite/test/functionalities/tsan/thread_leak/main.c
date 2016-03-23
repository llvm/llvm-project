//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

void *f1(void *p) {
    printf("hello\n");
    return NULL;
}

int main (int argc, char const *argv[])
{
    pthread_t t1;
    pthread_create(&t1, NULL, f1, NULL);

    return 0;
}
