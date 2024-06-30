// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x hip -disable-O0-optnone -emit-llvm -fcuda-is-device \
// RUN:   %s -o - | opt -S -passes=mem2reg | FileCheck %s

#define __device__ __attribute__((device))

typedef _Float16 half2 __attribute__((ext_vector_type(2)));
typedef _Float16 half4 __attribute__((ext_vector_type(4)));

typedef __bf16 bfloat2 __attribute__((ext_vector_type(2)));
typedef __bf16 bfloat4 __attribute__((ext_vector_type(4)));

// CHECK-LABEL: define dso_local noundef <2 x bfloat> @_Z40test_convertvector_from_half2_to_bfloat2Dv2_DF16_
// CHECK-SAME: (<2 x half> noundef [[IN:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[IN_ADDR:%.*]] = alloca <2 x half>, align 4, addrspace(5)
// CHECK-NEXT:    [[IN_ADDR_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[IN_ADDR]] to ptr
// CHECK-NEXT:    store <2 x half> [[IN]], ptr [[IN_ADDR_ASCAST]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x half>, ptr [[IN_ADDR_ASCAST]], align 4
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <2 x half> [[TMP0]] to <2 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <2 x float> [[FPEXT]] to <2 x bfloat>
// CHECK-NEXT:    ret <2 x bfloat> [[FPTRUNC]]
//
__device__ bfloat2 test_convertvector_from_half2_to_bfloat2(half2 in) {
  return __builtin_convertvector(in, bfloat2);
}

// CHECK-LABEL: define dso_local noundef <2 x half> @_Z40test_convertvector_from_bfloat2_to_half2Dv2_DF16b
// CHECK-SAME: (<2 x bfloat> noundef [[IN:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[IN_ADDR:%.*]] = alloca <2 x bfloat>, align 4, addrspace(5)
// CHECK-NEXT:    [[IN_ADDR_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[IN_ADDR]] to ptr
// CHECK-NEXT:    store <2 x bfloat> [[IN]], ptr [[IN_ADDR_ASCAST]], align 4
// CHECK-NEXT:    [[TMP0:%.*]] = load <2 x bfloat>, ptr [[IN_ADDR_ASCAST]], align 4
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <2 x bfloat> [[TMP0]] to <2 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <2 x float> [[FPEXT]] to <2 x half>
// CHECK-NEXT:    ret <2 x half> [[FPTRUNC]]
//
__device__ half2 test_convertvector_from_bfloat2_to_half2(bfloat2 in) {
  return __builtin_convertvector(in, half2);
}

// CHECK-LABEL: define dso_local noundef <4 x bfloat> @_Z40test_convertvector_from_half4_to_bfloat4Dv4_DF16_
// CHECK-SAME: (<4 x half> noundef [[IN:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[IN_ADDR:%.*]] = alloca <4 x half>, align 8, addrspace(5)
// CHECK-NEXT:    [[IN_ADDR_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[IN_ADDR]] to ptr
// CHECK-NEXT:    store <4 x half> [[IN]], ptr [[IN_ADDR_ASCAST]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x half>, ptr [[IN_ADDR_ASCAST]], align 8
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <4 x half> [[TMP0]] to <4 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <4 x float> [[FPEXT]] to <4 x bfloat>
// CHECK-NEXT:    ret <4 x bfloat> [[FPTRUNC]]
//
__device__ bfloat4 test_convertvector_from_half4_to_bfloat4(half4 in) {
  return __builtin_convertvector(in, bfloat4);
}

// CHECK-LABEL: define dso_local noundef <4 x half> @_Z40test_convertvector_from_bfloat4_to_half4Dv4_DF16b
// CHECK-SAME: (<4 x bfloat> noundef [[IN:%.*]]) #[[ATTR0]] {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[IN_ADDR:%.*]] = alloca <4 x bfloat>, align 8, addrspace(5)
// CHECK-NEXT:    [[IN_ADDR_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[IN_ADDR]] to ptr
// CHECK-NEXT:    store <4 x bfloat> [[IN]], ptr [[IN_ADDR_ASCAST]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load <4 x bfloat>, ptr [[IN_ADDR_ASCAST]], align 8
// CHECK-NEXT:    [[FPEXT:%.*]] = fpext <4 x bfloat> [[TMP0]] to <4 x float>
// CHECK-NEXT:    [[FPTRUNC:%.*]] = fptrunc <4 x float> [[FPEXT]] to <4 x half>
// CHECK-NEXT:    ret <4 x half> [[FPTRUNC]]
//
__device__ half4 test_convertvector_from_bfloat4_to_half4(bfloat4 in) {
  return __builtin_convertvector(in, half4);
}
