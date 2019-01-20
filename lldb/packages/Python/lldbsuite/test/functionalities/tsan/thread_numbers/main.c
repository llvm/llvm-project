//===-- main.c --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

char *pointer;

void *nothing(void *p) {
    return NULL;
}

void *f1(void *p) {
    pointer[0] = 'x';
    sleep(100);
    return NULL;
}

void *f2(void *p) {
    pointer[0] = 'y';
    sleep(100);
    return NULL;
}

int main (int argc, char const *argv[])
{
    pointer = (char *)malloc(10);

    for (int i = 0; i < 3; i++) {
        pthread_t t;
        pthread_create(&t, NULL, nothing, NULL);
        pthread_join(t, NULL);
    }

    pthread_t t1;
    pthread_create(&t1, NULL, f1, NULL);

    for (int i = 0; i < 3; i++) {
        pthread_t t;
        pthread_create(&t, NULL, nothing, NULL);
        pthread_join(t, NULL);
    }

    pthread_t t2;
    pthread_create(&t2, NULL, f2, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    return 0;
}
