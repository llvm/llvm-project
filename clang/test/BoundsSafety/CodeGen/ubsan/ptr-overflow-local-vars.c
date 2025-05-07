

// RUN: %clang_cc1 -O0 -fbounds-safety -Wno-int-conversion -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0 -fbounds-safety -Wno-int-conversion -x objective-c -fexperimental-bounds-safety-objc -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

int local_vars() {
  int * __bidi_indexable ptr = 0;
  return ptr += 1;
}

// CHECK:   call void @llvm.ubsantrap(i8 19)
