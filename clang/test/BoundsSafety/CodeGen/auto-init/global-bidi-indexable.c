
// RUN: %clang_cc1 -S -fbounds-safety -O0 -triple arm64-apple-darwin %s -o - | FileCheck %s
// RUN: %clang_cc1 -S -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -O0 -triple arm64-apple-darwin %s -o - | FileCheck %s

#include <ptrcheck.h>

int array1[1];
int array10[10];

int *__bidi_indexable bidi_ptr10 = array10;
// CHECK:	.globl	_bidi_ptr10 {{.*}} @bidi_ptr10
// ...
// CHECK:_bidi_ptr10:
// CHECK-NEXT:	.quad	_array10
// CHECK-NEXT:	.quad	_array10+40
// CHECK-NEXT:	.quad	_array10

int *__bidi_indexable bidi_ptr1 = array1;
// CHECK:	.globl	_bidi_ptr1 {{.*}} @bidi_ptr1
// ...
// CHECK:_bidi_ptr1:
// CHECK-NEXT:	.quad	_array1
// CHECK-NEXT:	.quad	_array1
// CHECK-NEXT:	.quad	_array1

int *__bidi_indexable bidi_ptr10_offset3 = array10 + 3;
// CHECK:	.globl	_bidi_ptr10_offset3 {{.*}} @bidi_ptr10_offset3
// ...
// CHECK: _bidi_ptr10_offset3:
// CHECK-NEXT:	.quad	_array10+12
// CHECK-NEXT:	.quad	_array10+40
// CHECK-NEXT:	.quad	_array10

int array_pp11_7[11][7];
int *__bidi_indexable bidi_ptrpp11_7_offset3 = array_pp11_7[3];
// CHECK:	.globl	_bidi_ptrpp11_7_offset3 {{.*}} @bidi_ptrpp11_7_offset3
// ...
// CHECK: _bidi_ptrpp11_7_offset3:
// CHECK-NEXT:  .quad	_array_pp11_7+84
// CHECK-NEXT:  .quad	_array_pp11_7+308
// CHECK-NEXT:  .quad	_array_pp11_7
