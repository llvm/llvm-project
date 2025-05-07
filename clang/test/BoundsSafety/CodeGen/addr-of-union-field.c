
// RUN: %clang_cc1 -O0  -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

union U {
    int i;
    long l;
};

// CHECK-LABEL: @foo
// CHECK: ret i32 1
int foo(void) {
    union U u;
    int *i = &u.i;
    long *l = &u.l;
    return (void *)i == (void *)l;
}
