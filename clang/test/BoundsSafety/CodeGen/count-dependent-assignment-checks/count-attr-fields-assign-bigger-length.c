

// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -emit-llvm %s -o /dev/null
// Executable version of "count-attr-fields-assign.c"
#include <ptrcheck.h>

struct S {
    int *__counted_by(l) bp;
    int *bp2 __counted_by(l+1);
    int l;
};

int foo();

int main () {
    int arr[10];
    struct S s = {arr, arr, 0};
    int n = 10;
    s.bp = &arr[1];
    s.bp2 = arr;
    s.l = n; // trap : s.1 + 1 >= countof(arr or s.bp2)

    return 0;
}
