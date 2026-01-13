// RUN: %clang_cc1 %s -cl-std=cl3.0 -emit-llvm -o - -triple spir-unknown-unknown | FileCheck %s
// RUN: %clang_cc1 %s -cl-std=cl3.0 -emit-llvm -o - -triple spirv64-unknown-unknown | FileCheck %s

kernel void test(global __bf16 *a, global __bf16 *b){
// CHECK-LABEL: spir_kernel void @test(
// CHECK: fadd bfloat
// CHECK: fsub bfloat
// CHECK: fmul bfloat
// CHECK: fdiv bfloat

  *b += *a;
  *b -= *a;
  *b *= *a;
  *b /= *a;
}

typedef __bf16 __bf16v4 __attribute((ext_vector_type(4)));

kernel void test_v4(global __bf16v4 *a, global __bf16v4 *b){
// CHECK-LABEL: spir_kernel void @test_v4(
// CHECK: fadd <4 x bfloat>
// CHECK: fsub <4 x bfloat>
// CHECK: fmul <4 x bfloat>
// CHECK: fdiv <4 x bfloat>

  *b += *a;
  *b -= *a;
  *b *= *a;
  *b /= *a;
}

