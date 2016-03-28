// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: MallocStackLogging=1 %run %t 2>&1 | FileCheck %s
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

void *foo(void *p) {
    return NULL;
}

int main() {
    pthread_t t;
    pthread_create(&t, NULL, foo, NULL);
    pthread_join(t, NULL);
    fprintf(stderr, "Done.\n");
    return 0;
}

// CHECK: Done.
