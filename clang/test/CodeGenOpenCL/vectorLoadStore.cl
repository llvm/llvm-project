// RUN: %clang_cc1 %s -emit-llvm -O0 -o - | FileCheck %s

typedef char char3 __attribute((ext_vector_type(3)));;

// Check for optimized vec3 load/store which treats vec3 as vec4.
void foo(char3 *P, char3 *Q) {
  *P = *Q;
  // CHECK: %{{.*}} = shufflevector <4 x i8> %{{.*}}, <4 x i8> undef, <3 x i32> <i32 0, i32 1, i32 2>
}
