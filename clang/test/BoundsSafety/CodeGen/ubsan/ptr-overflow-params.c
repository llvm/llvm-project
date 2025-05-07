

// RUN: %clang_cc1 -O0 -fbounds-safety -Wno-int-conversion -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O2 -fbounds-safety -Wno-int-conversion -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0 -fbounds-safety -Wno-int-conversion -x objective-c -fexperimental-bounds-safety-objc -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O2 -fbounds-safety -Wno-int-conversion -x objective-c -fexperimental-bounds-safety-objc -fsanitize=pointer-overflow -fsanitize-trap=pointer-overflow -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

int fixed_len_array(int * __bidi_indexable ptr, int k) {
  return ptr += k;
}

// CHECK:   call void @llvm.ubsantrap(i8 19)
