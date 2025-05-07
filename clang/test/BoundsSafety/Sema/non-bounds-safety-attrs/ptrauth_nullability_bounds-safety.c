

// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -fsyntax-only -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsyntax-only -ast-dump %s | FileCheck %s
#include <ptrcheck.h>

int * _Nullable __ptrauth(2, 0, 0) __single global_var1;
// CHECK: global_var1 'int *__single _Nullable __ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

int * _Nullable __single __ptrauth(2, 0, 0) global_var2;
// CHECK: global_var2 'int *__single _Nullable __ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

int * __ptrauth(2, 0, 0) _Nullable __single global_var3;
// CHECK: global_var3 'int *__single__ptrauth(2,0,0) _Nullable':'int *__single__ptrauth(2,0,0)'

int * __ptrauth(2, 0, 0) __single _Nullable global_var4;
// CHECK: global_var4 'int *__single__ptrauth(2,0,0) _Nullable':'int *__single__ptrauth(2,0,0)'

int * __single __ptrauth(2, 0, 0) _Nullable global_var5;
// CHECK: global_var5 'int *__single__ptrauth(2,0,0) _Nullable':'int *__single__ptrauth(2,0,0)'

int * __single _Nullable __ptrauth(2, 0, 0) global_var6;
// CHECK: global_var6 'int *__single _Nullable __ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

void foo() {
    int n1 = 0;
    int *_Nullable __ptrauth(2, 0, 0) __counted_by(n1) local_buf11;
    // CHECK: local_buf11 'int *__single __counted_by(n1) _Nullable __ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

    int n2 = 0;
    int *__ptrauth(2, 0, 0) _Nullable __counted_by(n2) local_buf12;
    // CHECK: local_buf12 'int *__single __counted_by(n2)__ptrauth(2,0,0) _Nullable':'int *__single__ptrauth(2,0,0)'

    int n3 = 0;
    int *__ptrauth(2, 0, 0) __counted_by(n3) _Nullable local_buf13;
    // CHECK: local_buf13 'int *__single __counted_by(n3)__ptrauth(2,0,0) _Nullable':'int *__single__ptrauth(2,0,0)'

    int n4 = 0;
    int *_Nullable __counted_by(n4) __ptrauth(2, 0, 0) local_buf21;
    // CHECK: local_buf21 'int *__single __counted_by(n4) _Nullable __ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

    int n5;
    int *__counted_by(n5) _Nullable __ptrauth(2, 0, 0) local_buf22;
    // CHECK: local_buf22 'int *__single __counted_by(n5) _Nullable __ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

    int n6;
    int *__counted_by(n6) __ptrauth(2, 0, 0) _Nullable local_buf23;
    // CHECK: local_buf23 'int *__single __counted_by(n6)__ptrauth(2,0,0) _Nullable':'int *__single__ptrauth(2,0,0)'

    int n7;
    int *_Nullable __ptrauth(2, 0, 0) local_buf31 __counted_by(n7);
    // CHECK: local_buf31 'int *__single __counted_by(n7) _Nullable __ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

    int n8;
    int *__ptrauth(2, 0, 0) _Nullable local_buf32 __counted_by(n8);
    // CHECK: local_buf32 'int *__single __counted_by(n8)__ptrauth(2,0,0) _Nullable':'int *__single__ptrauth(2,0,0)'
}
