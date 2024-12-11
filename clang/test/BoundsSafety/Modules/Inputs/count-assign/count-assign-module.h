#include <ptrcheck.h>

#ifndef COUNT_ASSIGN_MODULE_H
#define COUNT_ASSIGN_MODULE_H

int foo(int *__counted_by(len) ptr, int len) {
    int arr[10];
    ptr = arr;
    len = 10;
    return ptr[len-1];
}

void baz() {
    int len;
    void *__sized_by(len) ptr;
}
#endif