

// RUN: %clang_cc1 -O0  -fbounds-safety %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety %s -o /dev/null
// RUN: %clang_cc1 -O0  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s -o /dev/null
#include <ptrcheck.h>

int main() {
    int len;
    int *__counted_by(len) ptr1;
    int len2;
    int *__counted_by(len2) ptr2;
    unsigned long long diff = (unsigned long long)(ptr1 - ptr2);

    return 0;
}
