// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple   \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple   \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

// CHECK-LABEL: test_int
int test_int(int expr, uint idx) {
  // CHECK-SPIRV: %[[#entry_tok:]] = call token @llvm.experimental.convergence.entry()

  // CHECK-SPIRV:  %[[RET:.*]] = call [[TY:.*]] @llvm.spv.wave.read.lane.at.i32([[TY]] %[[#]], i32 %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.read.lane.at.i32([[TY]] %[[#]], i32 %[[#]])

  // CHECK:  ret [[TY]] %[[RET]]
  return WaveReadLaneAt(expr, idx);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.read.lane.at.i32([[TY]], i32) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.read.lane.at.i32([[TY]], i32) #[[#attr:]]

// Test basic lowering to runtime function call with array and float value.

// CHECK-LABEL: test_floatv4
float4 test_floatv4(float4 expr, uint idx) {
  // CHECK-SPIRV: %[[#entry_tok1:]] = call token @llvm.experimental.convergence.entry()

  // CHECK-SPIRV:  %[[RET1:.*]] = call [[TY1:.*]] @llvm.spv.wave.read.lane.at.v4f32([[TY1]] %[[#]], i32 %[[#]])
  // CHECK-DXIL:  %[[RET1:.*]] = call [[TY1:.*]] @llvm.dx.wave.read.lane.at.v4f32([[TY1]] %[[#]], i32 %[[#]])

  // CHECK:  ret [[TY1]] %[[RET1]]
  return WaveReadLaneAt(expr, idx);
}

// CHECK-DXIL: declare [[TY1]] @llvm.dx.wave.read.lane.at.v4f32([[TY1]], i32) #[[#attr]]
// CHECK-SPIRV: declare [[TY1]] @llvm.spv.wave.read.lane.at.v4f32([[TY1]], i32) #[[#attr]]

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
