// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

// CHECK-LABEL: test_int
int test_int(int expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.reduce.sum.i32([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.reduce.sum.i32([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveSum(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.reduce.sum.i32([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare spir_func [[TY]] @llvm.spv.wave.reduce.sum.i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint64_t
uint64_t test_uint64_t(uint64_t expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.reduce.sum.i64([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.reduce.usum.i64([[TY]] %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveActiveSum(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.reduce.usum.i64([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare spir_func [[TY]] @llvm.spv.wave.reduce.sum.i64([[TY]]) #[[#attr:]]

// Test basic lowering to runtime function call with array and float value.

// CHECK-LABEL: test_floatv4
float4 test_floatv4(float4 expr) {
  // CHECK-SPIRV:  %[[RET1:.*]] = call reassoc nnan ninf nsz arcp afn spir_func [[TY1:.*]] @llvm.spv.wave.reduce.sum.v4f32([[TY1]] %[[#]]
  // CHECK-DXIL:  %[[RET1:.*]] = call reassoc nnan ninf nsz arcp afn [[TY1:.*]] @llvm.dx.wave.reduce.sum.v4f32([[TY1]] %[[#]])
  // CHECK:  ret [[TY1]] %[[RET1]]
  return WaveActiveSum(expr);
}

// CHECK-DXIL: declare [[TY1]] @llvm.dx.wave.reduce.sum.v4f32([[TY1]]) #[[#attr]]
// CHECK-SPIRV: declare spir_func [[TY1]] @llvm.spv.wave.reduce.sum.v4f32([[TY1]]) #[[#attr]]

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
