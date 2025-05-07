

// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -fsyntax-only -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsyntax-only -ast-dump %s | FileCheck %s

#include <ptrcheck.h>

int * _Nullable __ptrauth(2, 0, 0) global_var1;
// CHECK: global_var1 'int *__single _Nullable __ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

int * __ptrauth(2, 0, 0) _Nullable global_var2;
// CHECK: global_var2 'int *__single__ptrauth(2,0,0) _Nullable':'int *__single__ptrauth(2,0,0)'
