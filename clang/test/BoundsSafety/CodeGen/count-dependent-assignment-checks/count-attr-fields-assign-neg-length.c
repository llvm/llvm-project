

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
    int n = -1;
    s.bp = &arr[1];
    s.bp2 = arr;
    s.l = n; // trap : no negative len

    return 0;
}

int bar () {
    int arr[10];
    struct S s = {arr, arr, 0};
    s.bp = &arr[1];
    s.bp2 = arr;
    s.l = 0; // no trap

    return 0;
}


// CHECK: define noundef i32 @foo()
// CHECK: {{.*}}:
// CHECK:   tail call void @llvm.ubsantrap(i8 25)
// CHECK:   unreachable
// CHECK: }

// CHECK: define noundef i32 @bar()
// CHECK: entry:
// CHECK:   ret i32 0
// CHECK: }
