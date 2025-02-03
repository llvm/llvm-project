// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -fnative-half-type -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -fnative-half-type -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call for int values.

// CHECK-LABEL: test_int
int test_int(int expr, uint idx) {
  // CHECK-SPIRV: %[[#entry_tok0:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.readlane.i32([[TY]] %[[#]], i32 %[[#]]) [ "convergencectrl"(token %[[#entry_tok0]]) ]
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.readlane.i32([[TY]] %[[#]], i32 %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveReadLaneAt(expr, idx);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.readlane.i32([[TY]], i32) #[[#attr:]]
// CHECK-SPIRV: declare spir_func [[TY]] @llvm.spv.wave.readlane.i32([[TY]], i32) #[[#attr:]]

#ifdef __HLSL_ENABLE_16_BIT
// CHECK-LABEL: test_int16
int16_t test_int16(int16_t expr, uint idx) {
  // CHECK-SPIRV: %[[#entry_tok1:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.readlane.i16([[TY]] %[[#]], i32 %[[#]]) [ "convergencectrl"(token %[[#entry_tok1]]) ]
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.readlane.i16([[TY]] %[[#]], i32 %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveReadLaneAt(expr, idx);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.readlane.i16([[TY]], i32) #[[#attr:]]
// CHECK-SPIRV: declare spir_func [[TY]] @llvm.spv.wave.readlane.i16([[TY]], i32) #[[#attr:]]
#endif

// Test basic lowering to runtime function call with array and float values.

// CHECK-LABEL: test_half
half test_half(half expr, uint idx) {
  // CHECK-SPIRV: %[[#entry_tok2:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV:  %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn spir_func [[TY:.*]] @llvm.spv.wave.readlane.f16([[TY]] %[[#]], i32 %[[#]]) [ "convergencectrl"(token %[[#entry_tok2]]) ]
  // CHECK-DXIL:  %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.dx.wave.readlane.f16([[TY]] %[[#]], i32 %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveReadLaneAt(expr, idx);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.readlane.f16([[TY]], i32) #[[#attr:]]
// CHECK-SPIRV: declare spir_func [[TY]] @llvm.spv.wave.readlane.f16([[TY]], i32) #[[#attr:]]

// CHECK-LABEL: test_double
double test_double(double expr, uint idx) {
  // CHECK-SPIRV: %[[#entry_tok3:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV:  %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn spir_func [[TY:.*]] @llvm.spv.wave.readlane.f64([[TY]] %[[#]], i32 %[[#]]) [ "convergencectrl"(token %[[#entry_tok3]]) ]
  // CHECK-DXIL:  %[[RET:.*]] = call reassoc nnan ninf nsz arcp afn [[TY:.*]] @llvm.dx.wave.readlane.f64([[TY]] %[[#]], i32 %[[#]])
  // CHECK:  ret [[TY]] %[[RET]]
  return WaveReadLaneAt(expr, idx);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.readlane.f64([[TY]], i32) #[[#attr:]]
// CHECK-SPIRV: declare spir_func [[TY]] @llvm.spv.wave.readlane.f64([[TY]], i32) #[[#attr:]]

// CHECK-LABEL: test_floatv4
float4 test_floatv4(float4 expr, uint idx) {
  // CHECK-SPIRV: %[[#entry_tok4:]] = call token @llvm.experimental.convergence.entry()
  // CHECK-SPIRV:  %[[RET1:.*]] = call reassoc nnan ninf nsz arcp afn spir_func [[TY1:.*]] @llvm.spv.wave.readlane.v4f32([[TY1]] %[[#]], i32 %[[#]]) [ "convergencectrl"(token %[[#entry_tok4]]) ]
  // CHECK-DXIL:  %[[RET1:.*]] = call reassoc nnan ninf nsz arcp afn [[TY1:.*]] @llvm.dx.wave.readlane.v4f32([[TY1]] %[[#]], i32 %[[#]])
  // CHECK:  ret [[TY1]] %[[RET1]]
  return WaveReadLaneAt(expr, idx);
}

// CHECK-DXIL: declare [[TY1]] @llvm.dx.wave.readlane.v4f32([[TY1]], i32) #[[#attr]]
// CHECK-SPIRV: declare spir_func [[TY1]] @llvm.spv.wave.readlane.v4f32([[TY1]], i32) #[[#attr]]

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
