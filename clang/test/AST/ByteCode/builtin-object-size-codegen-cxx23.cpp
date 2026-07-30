// RUN: %clang_cc1 -std=c++23 -fexperimental-new-constant-interpreter -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++23                                         -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

struct basic_filebuf {
  char __extbuf_;
  char __extbuf_min_[8];
};
// CHECK-LABEL: @_Z4swapR13basic_filebuf
void swap(basic_filebuf &__rhs) {
  int gi;
  // CHECK: store i32 8
  gi = __builtin_object_size(__rhs.__extbuf_min_, 0);
}


