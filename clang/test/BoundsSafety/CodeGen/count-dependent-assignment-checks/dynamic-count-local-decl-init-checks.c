

// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>
int foo () {
    int arr[10];
    int n = 11;
    int len = n; // trap : 11 > boundsof(arr)
    int *__counted_by(len) buf = arr;

    return 0;
}

int bar () {
    int arr[10];
    int len = 10;
    int *__counted_by(len) buf = arr;

    return 0;
}

// CHECK: define noundef i32 @foo()
// CHECK: {{.*}}:
// CHECK:   tail call void @llvm.ubsantrap(i8 25)
// CHECK:   unreachable

// CHECK: define noundef i32 @bar()
// CHECK: {{.*}}:
// CHECK:   ret i32 0
