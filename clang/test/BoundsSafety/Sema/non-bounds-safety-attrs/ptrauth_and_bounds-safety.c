

// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -fsyntax-only -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsyntax-only -ast-dump %s | FileCheck %s


#include <ptrcheck.h>

int * __ptrauth(2, 0, 0) __single global_var1;
// CHECK: global_var1 'int *__single__ptrauth(2,0,0)'

int * __single __ptrauth(2, 0, 0) global_var2;
// CHECK: global_var2 'int *__single__ptrauth(2,0,0)'

void foo(void) {
  int n;
  int *__ptrauth(2, 0, 0) __counted_by(n) local_buf1;
  // CHECK: local_buf1 'int *__single __counted_by(n)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

  int n2;
  int *__counted_by(n2) __ptrauth(2, 0, 0) local_buf2;
  // CHECK: local_buf2 'int *__single __counted_by(n2)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

  int n3;
  int *__ptrauth(2, 0, 0) local_buf3 __counted_by(n3);
  // CHECK: local_buf3 'int *__single __counted_by(n3)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

  int n4;
  unsigned char *__ptrauth(2, 0, 0) __counted_by(n4) local_byte_buf1;
  // CHECK: local_byte_buf1 'unsigned char *__single __counted_by(n4)__ptrauth(2,0,0)':'unsigned char *__single__ptrauth(2,0,0)'

  int n5;
  unsigned char *__counted_by(n5) __ptrauth(2, 0, 0) local_byte_buf2;
  // CHECK: local_byte_buf2 'unsigned char *__single __counted_by(n5)__ptrauth(2,0,0)':'unsigned char *__single__ptrauth(2,0,0)'

  int n6;
  unsigned char *__ptrauth(2, 0, 0) local_byte_buf3 __counted_by(n6);
  // CHECK: local_byte_buf3 'unsigned char *__single __counted_by(n6)__ptrauth(2,0,0)':'unsigned char *__single__ptrauth(2,0,0)'
}

struct Foo {
  int * __ptrauth(2, 0, 0) __single member_var1;
  // CHECK: member_var1 'int *__single__ptrauth(2,0,0)'

  int * __single __ptrauth(2, 0, 0) member_var2;
  // CHECK: member_var2 'int *__single__ptrauth(2,0,0)'

  int n;
  int *__ptrauth(2, 0, 0) __counted_by(n) member_buf1;
  // CHECK: member_buf1 'int *__single __counted_by(n)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

  int *__counted_by(n) __ptrauth(2, 0, 0) member_buf2;
  // CHECK: member_buf2 'int *__single __counted_by(n)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

  int *__ptrauth(2, 0, 0) member_buf3 __counted_by(n);
  // CHECK: member_buf3 'int *__single __counted_by(n)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

  unsigned char *__ptrauth(2, 0, 0) __counted_by(n) member_byte_buf1;
  // CHECK: member_byte_buf1 'unsigned char *__single __counted_by(n)__ptrauth(2,0,0)':'unsigned char *__single__ptrauth(2,0,0)'

  unsigned char *__counted_by(n) __ptrauth(2, 0, 0) member_byte_buf2;
  // CHECK: member_byte_buf2 'unsigned char *__single __counted_by(n)__ptrauth(2,0,0)':'unsigned char *__single__ptrauth(2,0,0)'

  unsigned char *__ptrauth(2, 0, 0) member_byte_buf3 __counted_by(n);
  // CHECK: member_byte_buf3 'unsigned char *__single __counted_by(n)__ptrauth(2,0,0)':'unsigned char *__single__ptrauth(2,0,0)'
};

union U {
  int * __ptrauth(2, 0, 0) __single union_var1;
  // CHECK: union_var1 'int *__single__ptrauth(2,0,0)'

  int * __single __ptrauth(2, 0, 0) union_var2;
  // CHECK: union_var2 'int *__single__ptrauth(2,0,0)'
};

void bar(void) {
  int n;
  int *__ptrauth(2, 0, 0) __counted_by(n) local_buf1;
  // CHECK: local_buf1 'int *__single __counted_by(n)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'
  int n2;
  int *__counted_by(n2) __ptrauth(2, 0, 0) local_buf2;
  // CHECK: local_buf2 'int *__single __counted_by(n2)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

  int n3;
  int *__ptrauth(2, 0, 0) local_buf3 __counted_by(n3);
  // CHECK: local_buf3 'int *__single __counted_by(n3)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'
}

struct Bar {
  int n;
  int *__ptrauth(2, 0, 0) __counted_by(n) member_buf1;
  // CHECK: member_buf1 'int *__single __counted_by(n)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

  int n2;
  int *__counted_by(n2) __ptrauth(2, 0, 0) member_buf2;
  // CHECK: member_buf2 'int *__single __counted_by(n2)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'

  int n3;
  int *__ptrauth(2, 0, 0) member_buf3 __counted_by(n3);
  // CHECK: member_buf3 'int *__single __counted_by(n3)__ptrauth(2,0,0)':'int *__single__ptrauth(2,0,0)'
};
