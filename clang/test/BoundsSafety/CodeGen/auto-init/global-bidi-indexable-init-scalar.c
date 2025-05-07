
// RUN: %clang_cc1 -S -fbounds-safety -triple arm64-apple-darwin -O0 %s -o - | FileCheck %s
// RUN: %clang_cc1 -S -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -triple arm64-apple-darwin -O0 %s -o - | FileCheck %s

#include <ptrcheck.h>

char a;
int *__bidi_indexable ptr = &a;
// CHECK: .globl	_ptr {{.*}} @ptr
// ...
// CHECK: _ptr:
// CHECK-NEXT:	.quad	_a
// CHECK-NEXT:	.quad	_a+1
// CHECK-NEXT:	.quad	_a
