

// RUN: %clang_cc1 -triple x86_64 -O0 -fbounds-safety -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64 -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

int array[3] = {42, 43, 44};

int* __bidi_indexable bidi_ptr = array;

// CHECK: %"[[BSS_BIDI_STRUCT:.*]]" = type { ptr, ptr, ptr }

// CHECK: [[ARRAY:.*]] =
// CHECK: {{.*}} = {{.*}} %"[[BSS_BIDI_STRUCT]]" { ptr @array, ptr inttoptr (i64 add (i64 ptrtoint (ptr [[ARRAY]] to i64), i64 12) to ptr), ptr [[ARRAY]] }, align 8
