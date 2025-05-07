

// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fbounds-safety -x c++ -fexperimental-bounds-safety-cxx -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK:      FieldDecl {{.+}} ptr 'int *{{.*}}__counted_by(count)':'int *{{.*}}'
// CHECK-NEXT: FieldDecl {{.+}} referenced count 'int'
// CHECK-NEXT: `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
struct cb {
  int *__counted_by(count) ptr;
  int count;
};

// CHECK:      FieldDecl {{.+}} ptr 'int *{{.*}}__counted_by_or_null(count)':'int *{{.*}}'
// CHECK-NEXT: FieldDecl {{.+}} referenced count 'int'
// CHECK-NEXT: `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
struct cbn {
  int *__counted_by_or_null(count) ptr;
  int count;
};

// CHECK:      FieldDecl {{.+}} ptr 'void *{{.*}}__sized_by(size)':'void *{{.*}}'
// CHECK-NEXT: FieldDecl {{.+}} referenced size 'int'
// CHECK-NEXT: `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
struct sb {
  void *__sized_by(size) ptr;
  int size;
};

// CHECK:      FieldDecl {{.+}} ptr 'void *{{.*}}__sized_by_or_null(size)':'void *{{.*}}'
// CHECK-NEXT: FieldDecl {{.+}} referenced size 'int'
// CHECK-NEXT: `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
struct sbn {
  void *__sized_by_or_null(size) ptr;
  int size;
};
