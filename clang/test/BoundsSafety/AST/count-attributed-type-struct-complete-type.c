

// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fbounds-safety -x c++ -fexperimental-bounds-safety-cxx -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
#include <stddef.h>

// This test ensures that the parsing is done late enough to have a complete
// struct type and be able to use offsetof()/sizeof().

// CHECK:      RecordDecl {{.+}} struct foo
// CHECK:      FieldDecl {{.+}} dummy 'int'
// CHECK-NEXT: FieldDecl {{.+}} ptr 'int *{{.*}}__counted_by(0UL)':'int *{{.*}}'
// CHECK-NEXT: FieldDecl {{.+}} ptr2 'int *{{.*}}__counted_by(24UL)':'int *{{.*}}'
// CHECK-NEXT: FieldDecl {{.+}} dummy2 'int'
struct foo {
  int dummy;
  int *__counted_by(offsetof(struct foo, dummy)) ptr;
  int *__counted_by(offsetof(struct foo, dummy2)) ptr2;
  int dummy2;
};

// CHECK:      RecordDecl {{.+}} struct bar
// CHECK:      FieldDecl {{.+}} dummy 'int'
// CHECK-NEXT: FieldDecl {{.+}} ptr 'int *{{.*}}__counted_by(16UL)':'int *{{.*}}'
struct bar {
  int dummy;
  int *__counted_by(sizeof(struct bar)) ptr;
};
