// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fbounds-safety -x c++ -fexperimental-bounds-safety-cxx -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fexperimental-bounds-safety-attributes -x c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
#include <stddef.h>

// CHECK:      FunctionDecl {{.+}} memcmp 'int (const void *[[single:(__single)?]] __sized_by(n), const void *[[single]] __sized_by(n), size_t)'
// CHECK-NEXT: ParmVarDecl [[s1:0x[^ ]+]] {{.+}} s1 'const void *[[single]] __sized_by(n)':'const void *[[single]]'
// CHECK-NEXT: ParmVarDecl [[s2:0x[^ ]+]] {{.+}} s2 'const void *[[single]] __sized_by(n)':'const void *[[single]]'
// CHECK-NEXT: ParmVarDecl {{.+}} used n 'size_t':'unsigned long'
// CHECK-NEXT: `-DependerDeclsAttr {{.+}} Implicit [[s1]] [[s2]] 0 0
int memcmp(const void *__sized_by(n) s1, const void *__sized_by(n) s2, size_t n);

// CHECK:      FunctionDecl {{.+}} memcpy 'void *[[single]] __sized_by(n)(void *[[single]] __sized_by(n), const void *[[single]] __sized_by(n), size_t)'
// CHECK-NEXT: ParmVarDecl [[dst:0x[^ ]+]] {{.+}} dst 'void *[[single]] __sized_by(n)':'void *[[single]]'
// CHECK-NEXT: ParmVarDecl [[src:0x[^ ]+]] {{.+}} src 'const void *[[single]] __sized_by(n)':'const void *[[single]]'
// CHECK-NEXT: ParmVarDecl {{.+}} used n 'size_t':'unsigned long'
// CHECK-NEXT: `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit [[dst]] [[src]] 0 0
void *__sized_by(n) memcpy(void *__sized_by(n) dst, const void *__sized_by(n) src, size_t n);
