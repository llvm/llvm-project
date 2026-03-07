// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

// CHECK-LABEL: test_int
bool test_int(int expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func i1 @llvm.spv.wave.all.equal.i32(i32
  // CHECK-DXIL:  %[[RET:.*]] = call i1 @llvm.dx.wave.all.equal.i32(i32
  // CHECK:  ret i1 %[[RET]]
  return WaveActiveAllEqual(expr);
}

// CHECK-DXIL: declare i1 @llvm.dx.wave.all.equal.i32(i32) #[[attr:.*]]
// CHECK-SPIRV: declare i1 @llvm.spv.wave.all.equal.i32(i32) #[[attr:.*]]

// CHECK-LABEL: test_uint64_t
bool test_uint64_t(uint64_t expr) {
  // CHECK-SPIRV:  %[[RET:.*]] = call spir_func i1 @llvm.spv.wave.all.equal.i64(i64 
  // CHECK-DXIL:  %[[RET:.*]] = call i1 @llvm.dx.wave.all.equal.i64(i64
  // CHECK:  ret i1 %[[RET]]
  return WaveActiveAllEqual(expr);
}

// CHECK-DXIL: declare i1 @llvm.dx.wave.all.equal.i64(i64) #[[attr]]
// CHECK-SPIRV: declare i1 @llvm.spv.wave.all.equal.i64(i64) #[[attr]]

// Test basic lowering to runtime function call with array and float value.

// CHECK-LABEL: test_floatv4
bool4 test_floatv4(float4 expr) {
  // CHECK-SPIRV:  %[[RET1:.*]] = call spir_func <4 x i1> @llvm.spv.wave.all.equal.v4f32(<4 x float> 
  // CHECK-DXIL:  %[[RET1:.*]] = call <4 x i1> @llvm.dx.wave.all.equal.v4f32(<4 x float> 
  // CHECK:  ret <4 x i1> %[[RET1]]
  return WaveActiveAllEqual(expr);
}

// CHECK-DXIL: declare <4 x i1> @llvm.dx.wave.all.equal.v4f32(<4 x float>) #[[attr]]
// CHECK-SPIRV: declare <4 x i1> @llvm.spv.wave.all.equal.v4f32(<4 x float>) #[[attr]]

// CHECK: attributes #[[attr]] = {{{.*}} convergent {{.*}}}
