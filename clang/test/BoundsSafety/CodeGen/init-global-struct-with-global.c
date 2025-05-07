// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -O0 -fbounds-safety -S %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -S %s -o - | FileCheck %s

#include <ptrcheck.h>

struct Foo {
    int *__bidi_indexable inner;
    int *__indexable inner2;
};
int globalCh;

struct Foo global = {
    .inner = &globalCh,
    .inner2 = &globalCh
};

// CHECK: _global:
// CHECK-NEXT: 	.quad	_globalCh
// CHECK-NEXT: 	.quad	_globalCh+4
// CHECK-NEXT: 	.quad	_globalCh
// CHECK-NEXT: 	.quad	_globalCh
// CHECK-NEXT: 	.quad	_globalCh+4
