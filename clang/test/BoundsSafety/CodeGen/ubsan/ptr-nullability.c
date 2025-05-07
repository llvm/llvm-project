

// RUN: %clang_cc1 -O0 -fbounds-safety -fsanitize=nullability-arg,nullability-assign,nullability-return -fsanitize-trap=nullability-arg,nullability-assign,nullability-return -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -O0 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fsanitize=nullability-arg,nullability-assign,nullability-return -fsanitize-trap=nullability-arg,nullability-assign,nullability-return -emit-llvm %s -o - | FileCheck %s

#include <ptrcheck.h>

int* bidi_to_bidi(int * _Nonnull __bidi_indexable ptr, int * __bidi_indexable ptr2) {
  return ptr = ptr2;
}

// CHECK: ptr @bidi_to_bidi
// ...
// CHECK:  call void @llvm.ubsantrap(i8 22)

int* raw_to_idx(int * _Nonnull __indexable ptr, int * ptr2) {
  return ptr = ptr2;
}

// CHECK: ptr @raw_to_idx
// ...
// CHECK:  call void @llvm.ubsantrap(i8 22)

int* bidi_to_idx(int * _Nonnull __indexable ptr, int * __bidi_indexable ptr2) {
  return ptr = ptr2;
}

// CHECK: ptr @bidi_to_idx
// ...
// CHECK:  call void @llvm.ubsantrap(i8 22)


int* idx_to_bidi(int * _Nonnull __bidi_indexable ptr, int * __indexable ptr2) {
  return ptr = ptr2;
}

// CHECK: ptr @idx_to_bidi
// ...
// CHECK:  call void @llvm.ubsantrap(i8 22)
