// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -triple spir-unknown-unknown -fpreserve-vec3-type | FileCheck %s

typedef char char3 __attribute__((ext_vector_type(3)));
typedef char char8 __attribute__((ext_vector_type(8)));
typedef short short3 __attribute__((ext_vector_type(3)));
typedef double double2 __attribute__((ext_vector_type(2)));
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

void kernel float3_to_double2(global float3 *a, global double2 *b) {
  // CHECK-LABEL: spir_kernel void @float3_to_double2
  // CHECK: %[[LOAD_A:.*]] = load <3 x float>, <3 x float> addrspace(1)* %a, align 16
  // CHECK: %[[ASTYPE:.*]] = shufflevector <3 x float> %[[LOAD_A]], <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  // CHECK: %[[OUT_BC:.*]] = bitcast <2 x double> addrspace(1)* %b to <4 x float> addrspace(1)*
  // CHECK: store <4 x float> %[[ASTYPE]], <4 x float> addrspace(1)* %[[OUT_BC]], align 16
  *b = __builtin_astype(*a, double2);
}

void kernel char8_to_short3(global short3 *a, global char8 *b) {
  // CHECK-LABEL: spir_kernel void @char8_to_short3
  // CHECK: %[[IN_BC:.*]] = bitcast <8 x i8> addrspace(1)* %b to <4 x i16> addrspace(1)*
  // CHECK: %[[LOAD_B:.*]] = load <4 x i16>, <4 x i16> addrspace(1)* %[[IN_BC]]
  // CHECK: %[[ASTYPE:.*]] = shufflevector <4 x i16> %[[LOAD_B]], <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK: store <3 x i16> %[[ASTYPE]], <3 x i16> addrspace(1)* %a, align 8
  *a = __builtin_astype(*b, short3);
}

void from_char3(char3 a, global int *out) {
  // CHECK-LABEL: void @from_char3
  // CHECK: %[[ASTYPE:.*]] = shufflevector <3 x i8> %a, <3 x i8> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  // CHECK: %[[OUT_BC:.*]] = bitcast i32 addrspace(1)* %out to <4 x i8> addrspace(1)*
  // CHECK: store <4 x i8> %[[ASTYPE]], <4 x i8> addrspace(1)* %[[OUT_BC]]
  *out = __builtin_astype(a, int);
}

void from_short3(short3 a, global long *out) {
  // CHECK-LABEL: void @from_short3
  // CHECK: %[[ASTYPE:.*]] = shufflevector <3 x i16> %a, <3 x i16> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  // CHECK: %[[OUT_BC:.*]] = bitcast i64 addrspace(1)* %out to <4 x i16> addrspace(1)*
  // CHECK: store <4 x i16> %[[ASTYPE]], <4 x i16> addrspace(1)* %[[OUT_BC]]
  *out = __builtin_astype(a, long);
}

void scalar_to_char3(int a, global char3 *out) {
  // CHECK-LABEL: void @scalar_to_char3
  // CHECK: %[[IN_BC:.*]] = bitcast i32 %a to <4 x i8>
  // CHECK: %[[ASTYPE:.*]] = shufflevector <4 x i8> %[[IN_BC]], <4 x i8> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK: store <3 x i8> %[[ASTYPE]], <3 x i8> addrspace(1)* %out
  *out = __builtin_astype(a, char3);
}

void scalar_to_short3(long a, global short3 *out) {
  // CHECK-LABEL: void @scalar_to_short3
  // CHECK: %[[IN_BC:.*]] = bitcast i64 %a to <4 x i16>
  // CHECK: %[[ASTYPE:.*]] = shufflevector <4 x i16> %[[IN_BC]], <4 x i16> poison, <3 x i32> <i32 0, i32 1, i32 2>
  // CHECK: store <3 x i16> %[[ASTYPE]], <3 x i16> addrspace(1)* %out
  *out = __builtin_astype(a, short3);
}
