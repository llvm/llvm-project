

// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -emit-llvm %s -o - | FileCheck %s
#include <ptrcheck.h>

struct S {
    int len;
    int *__counted_by(len) ptr;
};

struct T {
    union {
        struct S s;
        int arr[2];
    } u;

    void *vptr;
};

int foo() {
    int arr[9] = {0};
    struct S s = {9, arr};
    struct S sa[10] = {s, {8, &arr[2]}, 0}; // trap

    return 0;
}

int bar() {
    int arr[9] = {0};
    struct S s = {9, arr};
    struct S sa[10] = {s, {7, &arr[2]}, 0}; // no trap

    return 0;
}

// CHECK: define noundef i32 @foo()
// CHECK: {{.*}}:
// CHECK:   tail call void @llvm.ubsantrap(i8 25)
// CHECK:   unreachable

// CHECK: define noundef i32 @bar()
// CHECK: {{.*}}:
// CHECK:   ret i32 0
