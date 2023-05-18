// RUN: %clang_cc1 -emit-llvm -o - -triple i386-apple-darwin %s | FileCheck %s
// PR11930

typedef char vec_t __attribute__ ((__ext_vector_type__ (8)));
void h() {
// CHECK: store <8 x i8>
  vec_t v(0);
}
