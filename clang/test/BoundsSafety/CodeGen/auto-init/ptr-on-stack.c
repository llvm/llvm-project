

// RUN: %clang_cc1 -emit-llvm -fbounds-safety -O0 -triple arm64 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -O0 -triple arm64 %s -o - | FileCheck %s

#include <ptrcheck.h>

void single(void) {
  // CHECK: [[TMP0:%.*]] = alloca ptr, align 8
  // CHECK: store ptr null, ptr [[TMP0:%.*]], align 8, !annotation ![[ANNOT_ZERO_INIT:[0-9]+]]
  int *__single foo__single;
}

void indexable(void) {
  // CHECK: [[TMP1:%.*]] = alloca %"__bounds_safety::wide_ptr.indexable", align 8
  // CHECK: call void @llvm.memset.p0.i64(ptr align 8 [[TMP1]], i8 0, i64 16, i1 false), !annotation ![[ANNOT_ZERO_INIT]]
  int *__indexable foo__indexable;
}

void bidi_indexable(void) {
  // CHECK: [[TMP2:%.*]] = alloca %"__bounds_safety::wide_ptr.bidi_indexable", align 8
  // CHECK: call void @llvm.memset.p0.i64(ptr align 8 [[TMP2]], i8 0, i64 24, i1 false), !annotation ![[ANNOT_ZERO_INIT]]
  int *__bidi_indexable foo__bidi_indexable;
}

void counted_by(void) {
  // CHECK: [[TMP3:%.*]] = alloca ptr, align 8
  // CHECK: store ptr null, ptr [[TMP3:%.*]], align 8, !annotation ![[ANNOT_ZERO_INIT]]
  int len;
  int *__single __counted_by(len) foo__single__counted_by;
}

// CHECK: ![[ANNOT_ZERO_INIT]] = !{!"bounds-safety-zero-init"}
