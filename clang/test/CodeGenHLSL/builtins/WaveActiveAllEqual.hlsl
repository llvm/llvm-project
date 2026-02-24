// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

// CHECK-LABEL: test_int
bool test_int(int expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func i1 @llvm.spv.wave.all.equal.i32([[TY]] %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call i1 @llvm.dx.wave.all.equal.i32([[TY]] %[[#]])
  // CHECK:  ret i1 %[[RET]]
  return WaveActiveAllEqual(expr);
}

// CHECK-DXIL: declare i1 @llvm.dx.wave.all.equal.i32([[TY]]) #[[#attr:]]
// CHECK-SPIRV: declare i1 @llvm.spv.wave.all.equal.i32([[TY]]) #[[#attr:]]

// CHECK-LABEL: test_uint64_t
bool test_uint64_t(uint64_t expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func i1 @llvm.spv.wave.all.equal.i64(i64 %[[#]])
  // CHECK-DXIL:  %[[RET:.*]] = call i1 @llvm.dx.wave.uproduct.i64(i64 %[[#]])
  // CHECK:  ret i1 %[[RET]]
  return WaveActiveAllEqual(expr);
}

// CHECK-DXIL: declare i1 @llvm.dx.wave.uproduct.i64(i64 #[[#attr:]]
// CHECK-SPIRV: declare i1 @llvm.spv.wave.all.equal.i64(i64) #[[#attr:]]

// Test basic lowering to runtime function call with array and float value.

// CHECK-LABEL: test_floatv4
bool test_floatv4(float4 expr) {
  // CHECK-SPIRV:  %[[RET1:.*]] = call reassoc nnan ninf nsz arcp afn spir_func i1 @llvm.spv.wave.all.equal.v4f32(i32 %[[#]]
  // CHECK-DXIL:  %[[RET1:.*]] = call reassoc nnan ninf nsz arcp afn i1 @llvm.dx.wave.all.equal.v4f32(i32 %[[#]])
  // CHECK:  ret [[TY1]] %[[RET1]]
  return WaveActiveAllEqual(expr);
}

// CHECK-DXIL: declare i1 @llvm.dx.wave.all.equal.v4f32(i32) #[[#attr]]
// CHECK-SPIRV: declare i1 @llvm.spv.wave.all.equal.v4f32(i32) #[[#attr]]

// CHECK: attributes #[[#attr]] = {{{.*}} convergent {{.*}}}
