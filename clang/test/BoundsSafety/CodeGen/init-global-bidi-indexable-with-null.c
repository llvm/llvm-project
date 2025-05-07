

// RUN: %clang_cc1 -triple x86_64 -O0 -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64 -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

int *__bidi_indexable bidi_ptr = 0;
int *__bidi_indexable bidi_ptr2 = (int *)(void *)0;

// CHECK: %[[BSS_STRUCT_NAME:.*]] = type { ptr, ptr, ptr }
// CHECK: @bidi_ptr = {{.*}}global %[[BSS_STRUCT_NAME]] zeroinitializer
// CHECK: @bidi_ptr2 = {{.*}}global %[[BSS_STRUCT_NAME]] zeroinitializer
