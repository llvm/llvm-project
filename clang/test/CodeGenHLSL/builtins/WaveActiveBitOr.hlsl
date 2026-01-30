// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -fnative-int16-type -fnative-half-type \
// RUN:   -fmath-errno -ffp-contract=on -fno-rounding-math -finclude-default-header \
// RUN:   -disable-llvm-passes -o - |  FileCheck %s --check-prefixes=CHECK,CHECK-DXIL

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -fnative-int16-type -fnative-half-type \
// RUN:   -fmath-errno -ffp-contract=on -fno-rounding-math -finclude-default-header \ 
// RUN:   -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

// CHECK-LABEL: test_int
int test_int(int expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.bit.or.i32([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.bit.or.i32([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.bit.or.i32([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.bit.or.i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_int2
int2 test_int2(int2 expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.bit.or.v2i32([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.bit.or.v2i32([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.bit.or.v2i32([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.bit.or.v2i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_int3
int3 test_int3(int3 expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.bit.or.v3i32([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.bit.or.v3i32([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.bit.or.v3i32([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.bit.or.v3i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_int4
int4 test_int4(int4 expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.bit.or.v4i32([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.bit.or.v4i32([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.bit.or.v4i32([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.bit.or.v4i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_int16
int16_t test_int16_t(int16_t expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.bit.or.i16([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.bit.or.i16([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK-DXIL: declare [[TY]] @llvm.dx.wave.bit.or.i16([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare [[TY]] @llvm.spv.wave.bit.or.i16([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_int64
int64_t test_int64_t(int64_t expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.bit.or.i64([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.bit.or.i64([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK-LABEL: test_uint
uint test_uint(uint expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.bit.or.i32([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.bit.or.i32([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK-LABEL: test_uint16
uint16_t test_uint16_t(uint16_t expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.bit.or.i16([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.bit.or.i16([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK-LABEL: test_uint64
uint64_t test_uint64_t(uint64_t expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func [[TY:.*]] @llvm.spv.wave.bit.or.i64([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call [[TY:.*]] @llvm.dx.wave.bit.or.i64([[TY]] %[[#]])
  // CHECK: ret [[TY]] %[[RET]]
  return WaveActiveBitOr(expr);
}

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
