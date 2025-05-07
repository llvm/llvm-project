

// RUN: %clang_cc1 -emit-llvm -fbounds-safety -O0 -triple arm64 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -O0 -triple arm64 %s -o - | FileCheck %s

#include <ptrcheck.h>

int *__single foo__single;
int *__indexable foo__indexable;
int *__bidi_indexable foo__bidi_indexable;

// CHECK: @foo__single = global ptr null
// CHECK: @foo__indexable = global %"__bounds_safety::wide_ptr.indexable" zeroinitializer
// CHECK: @foo__bidi_indexable = global %"__bounds_safety::wide_ptr.bidi_indexable" zeroinitializer

int main(int argc, char **argv) { }
