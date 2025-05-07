
// RUN: %clang_cc1 -O0 -triple x86_64 -std=gnu99 -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0 -triple x86_64 -std=gnu99 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

// Check if 'noalias' attribute is added to thin pointers and dropped for wide pointers.

// __bidi_indexable

// CHECK-DAG: declare void @bidi_indexable_callee(ptr dead_on_unwind writable sret(%"__bounds_safety::wide_ptr.bidi_indexable") align 8, ptr noundef byval(%"__bounds_safety::wide_ptr.bidi_indexable") align 8)
int *__bidi_indexable bidi_indexable_callee(int *restrict __bidi_indexable p) __attribute__((malloc));

// CHECK-DAG: define dso_local void @bidi_indexable_caller(ptr dead_on_unwind noalias writable sret(%"__bounds_safety::wide_ptr.bidi_indexable") align 8 %agg.result, ptr noundef byval(%"__bounds_safety::wide_ptr.bidi_indexable") align 8 %p)
int *__bidi_indexable bidi_indexable_caller(int *restrict __bidi_indexable p) {
  // CHECK-DAG: call void @bidi_indexable_callee(ptr dead_on_unwind writable sret(%"__bounds_safety::wide_ptr.bidi_indexable") align 8 [[_:.*]], ptr noundef byval(%"__bounds_safety::wide_ptr.bidi_indexable") align 8 [[_:.*]])
  return bidi_indexable_callee(p);
}

// __indexable

// CHECK-DAG: declare { ptr, ptr } @indexable_callee(ptr noundef, ptr noundef)
int *__indexable indexable_callee(int *restrict __indexable p) __attribute__((malloc));

// CHECK-DAG: define dso_local { ptr, ptr } @indexable_caller(ptr noundef %p.coerce0, ptr noundef %p.coerce1)
int *__indexable indexable_caller(int *restrict __indexable p) {
  // CHECK-DAG: call { ptr, ptr } @indexable_callee(ptr noundef [[_:.*]], ptr noundef [[_:.*]])
  return indexable_callee(p);
}

// __single

// CHECK-DAG: declare noalias ptr @single_callee(ptr noundef)
int *__single single_callee(int *restrict __single p) __attribute__((malloc));

// CHECK-DAG: define dso_local ptr @single_caller(ptr noalias noundef %p)
int *__single single_caller(int *restrict __single p) {
  // CHECK-DAG: call noalias ptr @single_callee(ptr noundef [[_:.*]])
  return single_callee(p);
}

// __unsafe_indexable

// CHECK-DAG: declare noalias ptr @unsafe_indexable_callee(ptr noundef)
int *__unsafe_indexable unsafe_indexable_callee(int *restrict __unsafe_indexable p) __attribute__((malloc));

// CHECK-DAG: define dso_local ptr @unsafe_indexable_caller(ptr noalias noundef %p)
int *__unsafe_indexable unsafe_indexable_caller(int *restrict __unsafe_indexable p) {
  // CHECK-DAG: call noalias ptr @unsafe_indexable_callee(ptr noundef [[_:.*]])
  return unsafe_indexable_callee(p);
}

// __counted_by

// CHECK-DAG: declare noalias ptr @counted_by_callee(ptr noundef)
int *__counted_by(8) counted_by_callee(int *restrict __counted_by(8) p) __attribute__((malloc));

// CHECK-DAG: define dso_local ptr @counted_by_caller(ptr noalias noundef %p)
int *__counted_by(8) counted_by_caller(int *restrict __counted_by(8) p) {
  // CHECK-DAG: call noalias ptr @counted_by_callee(ptr noundef [[_:.*]])
  return counted_by_callee(p);
}
