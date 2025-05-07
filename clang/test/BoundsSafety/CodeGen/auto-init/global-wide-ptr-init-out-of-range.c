
// RUN: %clang_cc1 -S -fbounds-safety -O0 -triple arm64-apple-darwin %s -o - | FileCheck %s
// RUN: %clang_cc1 -S -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -O0 -triple arm64-apple-darwin %s -o - | FileCheck %s

#include <ptrcheck.h>

int array10[10];

int *__attribute__((bidi_indexable)) bidi0 = array10 + 10;

// CHECK: _bidi0:
// CHECK: 	.quad	_array10+40
// CHECK: 	.quad	_array10+40
// CHECK: 	.quad	_array10
