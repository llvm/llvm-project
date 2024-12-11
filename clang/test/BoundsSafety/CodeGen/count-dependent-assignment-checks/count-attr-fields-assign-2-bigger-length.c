

// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -emit-llvm %s -o /dev/null
// Executable version of "count-attr-fields-assign-2.c"
#include <ptrcheck.h>

struct S {
    int *__counted_by(l) bp;
    int *bp2 __counted_by(l+1);
    int l;
};

int main () {
    int arr[10];
    struct S s = {arr, arr, 0};
    int n = 10;
    s.bp = &arr[1];
    s.l = n; // run-time trap - s.l cannot be bigger than element count of &arr[1] (9) and s.1 + 1 cannot be bigger than element count of arr (10)
    s.bp2 = arr;

    return 0;
}
