
// RUN: %clang_cc1 -S -fbounds-safety -O0 -triple arm64-apple-darwin %s -o - | FileCheck %s
// RUN: %clang_cc1 -S -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -O0 -triple arm64-apple-darwin %s -o - | FileCheck %s

#include <ptrcheck.h>

int array0[0];
int *__attribute__((bidi_indexable)) bidi_ptr0 = array0;

// CHECK: _bidi_ptr0:
// CHECK: 	.quad	_array0
// CHECK: 	.quad	_array0
// CHECK: 	.quad	_array0
