// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown -fpreserve-vec3-type | FileCheck %s

typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));

void kernel foo(global float3 *a, global float3 *b) {
  // CHECK-LABEL: spir_kernel void @foo
  // CHECK: %[[LOAD_A:.*]] = load <3 x float>, <3 x float> addrspace(1)* %a
  // CHECK: store <3 x float> %[[LOAD_A]], <3 x float> addrspace(1)* %b
  *b = *a;
}

void kernel float4_to_float3(global float3 *a, global float4 *b) {
  // CHECK-LABEL: spir_kernel void @float4_to_float3
  // CHECK: %[[LOAD_A:.*]] = load <4 x float>, <4 x float> addrspace(1)* %b, align 16
  // CHECK: %[[ASTYPE:.*]] = shufflevector <4 x float> %[[LOAD_A]], <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK: store <3 x float> %[[ASTYPE]], <3 x float> addrspace(1)* %a, align 16
  *a = __builtin_astype(*b, float3);
}

void kernel float3_to_float4(global float3 *a, global float4 *b) {
  // CHECK-LABEL: spir_kernel void @float3_to_float4
  // CHECK: %[[LOAD_A:.*]] = load <3 x float>, <3 x float> addrspace(1)* %a, align 16
  // CHECK: %[[ASTYPE:.*]] = shufflevector <3 x float> %[[LOAD_A]], <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  // CHECK: store <4 x float> %[[ASTYPE]], <4 x float> addrspace(1)* %b, align 16
  *b = __builtin_astype(*a, float4);
}
