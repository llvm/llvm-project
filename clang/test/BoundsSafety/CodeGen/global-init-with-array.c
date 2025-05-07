// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -O0 -fbounds-safety -S %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx11.0.0 -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -S %s -o - | FileCheck %s

#include <ptrcheck.h>

int array[100][100];
int* __bidi_indexable ptr = array;

// CHECK: _ptr:
// CHECK: 	.quad	_array
// CHECK: 	.quad	_array+40000
// CHECK: 	.quad	_array
