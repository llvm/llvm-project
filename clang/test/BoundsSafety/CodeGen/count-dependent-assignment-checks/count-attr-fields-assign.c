

// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

struct S {
    int *__counted_by(l) bp;
    int *bp2 __counted_by(l+1);
    int l;
};

int foo () {
    int arr[10];
    struct S s = {arr, arr, 0};
    s.bp = &arr[1];
    s.bp2 = arr;
    s.l = 9;
    // run-time check here

    for (int i = 0; i < s.l; ++i)
        s.bp[i] = i;

    return s.bp2[8] + s.bp[7]; // in bounds
}

int bar () {
    int arr[10];
    struct S s = {arr, arr, 0};
    int n = 10;
    s.bp = &arr[1];
    s.bp2 = arr;
    s.l = n;
    // run-time check here

    for (int i = 0; i < s.l; ++i)
        s.bp[i] = i;

    return s.bp2[8] + s.bp[7]; // in bounds
}

// CHECK: define noundef {{.*}}i32 @foo() {{.*}}
// CHECK-NEXT: {{.*}}:
// CHECK-NEXT:   ret i32 14

// CHECK: define noundef i32 @bar() {{.*}}
// CHECK-NEXT: {{.*}}:
// CHECK-NEXT:   tail call void @llvm.ubsantrap(i8 25)
// CHECK:   unreachable
