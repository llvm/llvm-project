

#include <ptrcheck.h>

// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -fsyntax-only -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsyntax-only -ast-dump %s | FileCheck %s

int * __ptrauth(2, 0, 0) global_var;
// CHECK: global_var 'int *__single__ptrauth(2,0,0)'

int * __ptrauth(2, 0, 0) global_array[5];
// CHECK: global_array 'int *__single__ptrauth(2,0,0)[5]'

void foo(void) {
  int * __ptrauth(2, 0, 0) local_var;
}
// CHECK: local_var 'int *__bidi_indexable__ptrauth(2,0,0)'

void foo2(void) {
  int * __ptrauth(2, 0, 0) local_array[5];
}
// CHECK: local_array 'int *__single__ptrauth(2,0,0)[5]'

struct Foo {
  int * __ptrauth(2, 0, 0) member_var;
};
// CHECK: member_var 'int *__single__ptrauth(2,0,0)'

struct Foo2 {
  int * __ptrauth(2, 0, 0) member_array[5];
};
// CHECK: member_array 'int *__single__ptrauth(2,0,0)[5]'

union U {
  int * __ptrauth(2, 0, 0) union_var;
};
// CHECK: union_var 'int *__single__ptrauth(2,0,0)'

union U2 {
  int * __ptrauth(2, 0, 0) union_array[5];
};
// CHECK: union_array 'int *__single__ptrauth(2,0,0)[5]'
