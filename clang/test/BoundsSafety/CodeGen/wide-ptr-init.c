

// RUN: %clang_cc1 -O0 -triple arm64 -fbounds-safety -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

// CHECK: %"__bounds_safety::wide_ptr.indexable" = type { ptr, ptr }
// CHECK: %"__bounds_safety::wide_ptr.bidi_indexable" = type { ptr, ptr, ptr }

// CHECK: @i_null = global %"__bounds_safety::wide_ptr.indexable{{.*}}" zeroinitializer, align 8
// CHECK: @bi_null = global %"__bounds_safety::wide_ptr.bidi_indexable{{.*}}" zeroinitializer, align 8
int *__indexable i_null = 0;
int *__bidi_indexable bi_null = 0;

// CHECK: @i_null_cast = global %"__bounds_safety::wide_ptr.indexable{{.*}}" zeroinitializer, align 8
// CHECK: @bi_null_cast = global %"__bounds_safety::wide_ptr.bidi_indexable{{.*}}" zeroinitializer, align 8
void *__indexable i_null_cast = (int *)0;
void *__bidi_indexable bi_null_cast = (int *)0;

// CHECK: @array = global [10 x i32] zeroinitializer, align 4
int array[10];

// CHECK: @i_array = global %"__bounds_safety::wide_ptr.indexable{{.*}}" { ptr @array, ptr inttoptr (i64 add (i64 ptrtoint (ptr @array to i64), i64 40) to ptr) }, align 8
// CHECK: @bi_array = global %"__bounds_safety::wide_ptr.bidi_indexable{{.*}}" { ptr @array, ptr inttoptr (i64 add (i64 ptrtoint (ptr @array to i64), i64 40) to ptr), ptr @array }, align 8
int *__indexable i_array = array;
int *__bidi_indexable bi_array = array;

// CHECK: @i_array_cast = global %"__bounds_safety::wide_ptr.indexable{{.*}}" { ptr @array, ptr inttoptr (i64 add (i64 ptrtoint (ptr @array to i64), i64 40) to ptr) }, align 8
// CHECK: @bi_array_cast = global %"__bounds_safety::wide_ptr.bidi_indexable{{.*}}" { ptr @array, ptr inttoptr (i64 add (i64 ptrtoint (ptr @array to i64), i64 40) to ptr), ptr @array }, align 8
void *__indexable i_array_cast = array;
void *__bidi_indexable bi_array_cast = array;

// CHECK: @x = global i32 0, align 4
int x;

// CHECK: @i_addrof = global %"__bounds_safety::wide_ptr.indexable{{.*}}" { ptr @x, ptr inttoptr (i64 add (i64 ptrtoint (ptr @x to i64), i64 4) to ptr) }, align 8
// CHECK: @bi_addrof = global %"__bounds_safety::wide_ptr.bidi_indexable{{.*}}" { ptr @x, ptr inttoptr (i64 add (i64 ptrtoint (ptr @x to i64), i64 4) to ptr), ptr @x }, align 8
int *__indexable i_addrof = &x;
int *__bidi_indexable bi_addrof = &x;

// CHECK: @i_addrof_cast = global %"__bounds_safety::wide_ptr.indexable{{.*}}" { ptr @x, ptr inttoptr (i64 add (i64 ptrtoint (ptr @x to i64), i64 4) to ptr) }, align 8
// CHECK: @bi_addrof_cast = global %"__bounds_safety::wide_ptr.bidi_indexable{{.*}}" { ptr @x, ptr inttoptr (i64 add (i64 ptrtoint (ptr @x to i64), i64 4) to ptr), ptr @x }, align 8
void *__indexable i_addrof_cast = &x;
void *__bidi_indexable bi_addrof_cast = &x;

struct foo {
  int x[100];
};

// CHECK: @f = global %struct.foo zeroinitializer, align 4
struct foo f = {};

struct bar {
  const void *__bidi_indexable p;
};

// CHECK: @b = global %struct.bar { %"__bounds_safety::wide_ptr.bidi_indexable{{.*}}" { ptr @f, ptr inttoptr (i64 add (i64 ptrtoint (ptr @f to i64), i64 400) to ptr), ptr @f } }, align 8
struct bar b = {
  .p = &f
};
