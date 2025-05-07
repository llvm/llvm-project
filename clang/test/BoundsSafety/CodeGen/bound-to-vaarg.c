// RUN: %clang_cc1 -O0  -fbounds-safety -triple x86_64-linux-gnu %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -triple x86_64-linux-gnu %s -o /dev/null
// RUN: %clang_cc1 -O0  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -triple x86_64-linux-gnu %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -triple x86_64-linux-gnu %s -o /dev/null

#include <ptrcheck.h>

int foo_vaarg(int num, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, num);

    int *__unsafe_indexable ptr = __builtin_va_arg(ap, int*);
    return *ptr;
}

int main() {
    int a = 0;
    int *ptr = &a;
    foo_vaarg(1, ptr);

    int len;
    int *__counted_by(len) countPtr;
    foo_vaarg(1, countPtr);

    return 0;
}
