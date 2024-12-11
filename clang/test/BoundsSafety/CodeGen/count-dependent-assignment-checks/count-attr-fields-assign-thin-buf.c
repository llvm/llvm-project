

// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

struct S {
    int *__counted_by(l) bp;
    int *bp2 __counted_by(l+1);
    int l;
};

int foo () {
    int arr[16] = {0};
    int *__single tp = arr;
    struct S s = {arr, arr, 0};
    s.bp = &arr[3];
    s.bp2 = arr;
    s.l = 9;
    // run-time check here

    for (int i = 0; i < s.l; ++i)
        s.bp2[i] = i;

    int l = 10;
    s.l = l;
    s.bp = tp; // trap: s.l > 1
    s.bp2 = s.bp2;

    return 0;
}

int bar () {
    int arr[16] = {0};
    int *__single tp = arr;
    struct S s = {arr, arr, 0};
    s.bp = &arr[3];
    s.bp2 = arr;
    s.l = 9;
    // run-time check here

    for (int i = 0; i < s.l; ++i)
        s.bp2[i] = i;

    s.l = 1;
    s.bp = tp;
    s.bp2 = s.bp2;

    return 0;
}

// CHECK: define noundef i32 @foo()
// CHECK: {{.*}}:
// CHECK:   call void @llvm.ubsantrap(i8 25)
// CHECK:   unreachable
// CHECK: }

// CHECK: define noundef i32 @bar()
// CHECK: {{.*}}:
// CHECK:   ret i32 0
// CHECK: }
