

// RUN: %clang_cc1 -triple x86_64 -O0 -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64 -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s
#include <ptrcheck.h>

extern float foo[];
// CHECK: @foo = external global [0 x float]

float *__indexable wide_f[] = {foo};
// CHECK: @wide_f = global [1 x %"__bounds_safety::wide_ptr.indexable"] [%"__bounds_safety::wide_ptr.indexable" { ptr @foo, ptr @foo }]

extern float bar[];
float bar[] = {1, 2, 3, 4};
// CHECK: @bar = global [4 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00]

float *__indexable wide_f2[] = {bar};
// CHECK: @wide_f2 = global [1 x %"__bounds_safety::wide_ptr.indexable"] [%"__bounds_safety::wide_ptr.indexable" { ptr @bar, ptr inttoptr (i64 add (i64 ptrtoint (ptr @bar to i64), i64 16) to ptr) }]

float *__bidi_indexable wide_f3[] = {foo, bar};
// CHECK: @wide_f3 = global [2 x %"__bounds_safety::wide_ptr.bidi_indexable"] [%"__bounds_safety::wide_ptr.bidi_indexable" { ptr @foo, ptr @foo, ptr @foo }, %"__bounds_safety::wide_ptr.bidi_indexable" { ptr @bar, ptr inttoptr (i64 add (i64 ptrtoint (ptr @bar to i64), i64 16) to ptr), ptr @bar }]

// CHECK-NOT: @foo
// CHECK-NOT: @wide_f
// CHECK-NOT: @bar
// CHECK-NOT: @wide_f2
// CHECK-NOT: @wide_f3
