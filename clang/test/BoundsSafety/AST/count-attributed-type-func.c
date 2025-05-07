

// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fbounds-safety -x c++ -fexperimental-bounds-safety-cxx -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-iphoneos -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK:      FunctionDecl {{.+}} cb_in_in 'void (int *{{.*}}__counted_by(count), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} ptr 'int *{{.*}}__counted_by(count)':'int *{{.*}}'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used count 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
void cb_in_in(int *__counted_by(count) ptr, int count);

// CHECK:      FunctionDecl {{.+}} cb_in_out 'void (int *{{.*}}__counted_by(*count), int *__single)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} ptr 'int *{{.*}}__counted_by(*count)':'int *{{.*}}'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used count 'int *__single'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit IsDeref {{.+}} 0
void cb_in_out(int *__counted_by(*count) ptr, int *__single count);

// CHECK:      FunctionDecl {{.+}} cb_out_in 'void (int *{{.*}}__counted_by(count)*__single, int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} ptr 'int *{{.*}}__counted_by(count)*__single'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used count 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 1
void cb_out_in(int *__counted_by(count) *__single ptr, int count);

// CHECK:      FunctionDecl {{.+}} cb_out_out 'void (int *{{.*}}__counted_by(*count)*__single, int *__single)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} ptr 'int *{{.*}}__counted_by(*count)*__single'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used count 'int *__single'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit IsDeref {{.+}} 1
void cb_out_out(int *__counted_by(*count) *__single ptr, int *__single count);

// CHECK:      FunctionDecl {{.+}} cbn 'void (int *{{.*}}__counted_by_or_null(count), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} ptr 'int *{{.*}}__counted_by_or_null(count)':'int *{{.*}}'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used count 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
void cbn(int *__counted_by_or_null(count) ptr, int count);

// CHECK:      FunctionDecl {{.+}} sb 'void (void *{{.*}}__sized_by(size), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} ptr 'void *{{.*}}__sized_by(size)':'void *{{.*}}'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used size 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
void sb(void *__sized_by(size) ptr, int size);

// CHECK:      FunctionDecl {{.+}} sbn 'void (void *{{.*}}__sized_by_or_null(size), int)'
// CHECK-NEXT: |-ParmVarDecl {{.+}} ptr 'void *{{.*}}__sized_by_or_null(size)':'void *{{.*}}'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used size 'int'
// CHECK-NEXT:   `-DependerDeclsAttr {{.+}} Implicit {{.+}} 0
void sbn(void *__sized_by_or_null(size) ptr, int size);

// CHECK:      FunctionDecl {{.+}} rcb 'int *{{.*}}__counted_by(count)(int)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used count 'int'
int *__counted_by(count) rcb(int count);

// CHECK:      FunctionDecl {{.+}} rcbn 'int *{{.*}}__counted_by_or_null(count)(int)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used count 'int'
int *__counted_by_or_null(count) rcbn(int count);

// CHECK:      FunctionDecl {{.+}} rsb 'void *{{.*}}__sized_by(size)(int)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used size 'int'
void *__sized_by(size) rsb(int size);

// CHECK:      FunctionDecl {{.+}} rsbn 'void *{{.*}}__sized_by_or_null(size)(int)'
// CHECK-NEXT: `-ParmVarDecl {{.+}} used size 'int'
void *__sized_by_or_null(size) rsbn(int size);
