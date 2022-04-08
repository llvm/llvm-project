// RUN: %clang_cc1 -no-opaque-pointers -cl-std=CL2.0 -triple "spir-unknown-unknown" %s -emit-llvm -O0 -o - | FileCheck %s

typedef char char2 __attribute((ext_vector_type(2)));
typedef char char3 __attribute((ext_vector_type(3)));
typedef char char8 __attribute((ext_vector_type(8)));
typedef float float4 __attribute((ext_vector_type(4)));

// Check for optimized vec3 load/store which treats vec3 as vec4.
void foo(char3 *P, char3 *Q) {
  *P = *Q;
  // CHECK: %{{.*}} = shufflevector <4 x i8> %{{.*}}, <4 x i8> poison, <3 x i32> <i32 0, i32 1, i32 2>
}

// CHECK: define{{.*}} spir_func void @alignment()
void alignment() {
  __private char2 data_generic[100];
  __private char8 data_private[100];

  // CHECK: %{{.*}} = load <4 x float>, <4 x float> addrspace(4)* %{{.*}}, align 2
  // CHECK: store <4 x float> %{{.*}}, <4 x float>* %{{.*}}, align 8
  ((private float4 *)data_private)[1] = ((float4 *)data_generic)[2];
}
