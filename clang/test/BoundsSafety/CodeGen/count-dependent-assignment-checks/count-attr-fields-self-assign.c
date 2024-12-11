

// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

struct S {
    int *__counted_by(l) bp;
    int *bp2 __counted_by(l+1);
    int l;
};

void side_effect();

int foo () {
    int arr[10];
    struct S s = {arr, arr, 0};
    s.bp = &arr[1];
    s.bp2 = arr;
    s.l = 9;

    side_effect();

    s.bp = s.bp;
    s.bp2 = &arr[8]; // trap : bound of &arr[8] < s.l + 1
    s.l = s.l;

    return 0;
}

int bar () {
    int arr[10];
    struct S s = {arr, arr, 0};
    s.bp = &arr[1];
    s.bp2 = arr;
    s.l = 9;

    side_effect();

    s.bp = s.bp;
    s.bp2 = &arr[0]; // no trap
    s.l = s.l;

    return 0;
}

// CHECK: define noundef i32 @foo()
// CHECK: {{.*}}:
// CHECK:   tail call void @llvm.ubsantrap(i8 25)
// CHECK:   unreachable
// CHECK: }

// CHECK: define noundef i32 @bar()
// CHECK: {{.*}}:
// CHECK:   ret i32 0
