// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan1.3-library -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,SPIRV

// -- Case 1: scalar * scalar -> scalar --

// CHECK-LABEL: test_scalar_mulf
// CHECK: [[A:%.*]] = load float, ptr %a.addr
// CHECK: [[B:%.*]] = load float, ptr %b.addr
// CHECK: %hlsl.mul = fmul {{.*}} float [[A]], [[B]]
// CHECK: ret float %hlsl.mul
export float test_scalar_mulf(float a, float b) { return mul(a, b); }

// CHECK-LABEL: test_scalar_muli
// CHECK: [[A:%.*]] = load i32, ptr %a.addr
// CHECK: [[B:%.*]] = load i32, ptr %b.addr
// CHECK: %hlsl.mul = mul i32 [[A]], [[B]]
// CHECK: ret i32 %hlsl.mul
export int test_scalar_muli(int a, int b) { return mul(a, b); }

// -- Case 2: scalar * vector -> vector --

// CHECK-LABEL: test_scalar_vec_mul
// CHECK: [[A:%.*]] = load float, ptr %a.addr
// CHECK: [[B:%.*]] = load <3 x float>, ptr %b.addr
// CHECK: %.splatinsert = insertelement <3 x float> poison, float [[A]], i64 0
// CHECK: %.splat = shufflevector <3 x float> %.splatinsert, <3 x float> poison, <3 x i32> zeroinitializer
// CHECK: %hlsl.mul = fmul {{.*}} <3 x float> %.splat, [[B]]
// CHECK: ret <3 x float> %hlsl.mul
export float3 test_scalar_vec_mul(float a, float3 b) { return mul(a, b); }

// -- Case 3: scalar * matrix -> matrix --

// CHECK-LABEL: test_scalar_mat_mul
// CHECK: [[A:%.*]] = load float, ptr %a.addr
// CHECK: [[B:%.*]] = load <6 x float>, ptr %b.addr
// CHECK: %.splatinsert = insertelement <6 x float> poison, float [[A]], i64 0
// CHECK: %.splat = shufflevector <6 x float> %.splatinsert, <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: %hlsl.mul = fmul {{.*}} <6 x float> %.splat, [[B]]
// CHECK: ret <6 x float> %hlsl.mul
export float2x3 test_scalar_mat_mul(float a, float2x3 b) { return mul(a, b); }

// -- Case 4: vector * scalar -> vector --

// CHECK-LABEL: test_vec_scalar_mul
// CHECK: [[A:%.*]] = load <3 x float>, ptr %a.addr
// CHECK: [[B:%.*]] = load float, ptr %b.addr
// CHECK: %.splatinsert = insertelement <3 x float> poison, float [[B]], i64 0
// CHECK: %.splat = shufflevector <3 x float> %.splatinsert, <3 x float> poison, <3 x i32> zeroinitializer
// CHECK: %hlsl.mul = fmul {{.*}} <3 x float> %.splat, [[A]]
// CHECK: ret <3 x float> %hlsl.mul
export float3 test_vec_scalar_mul(float3 a, float b) { return mul(a, b); }

// -- Case 5: vector * vector -> scalar (dot product) --

// CHECK-LABEL: test_vec_vec_mul
// CHECK: [[A:%.*]] = load <3 x float>, ptr %a.addr
// CHECK: [[B:%.*]] = load <3 x float>, ptr %b.addr
// DXIL: %hlsl.mul = call {{.*}} float @llvm.dx.fdot.v3f32(<3 x float> [[A]], <3 x float> [[B]])
// SPIRV: %hlsl.mul = call {{.*}} float @llvm.spv.fdot.v3f32(<3 x float> [[A]], <3 x float> [[B]])
// CHECK: ret float %hlsl.mul
export float test_vec_vec_mul(float3 a, float3 b) { return mul(a, b); }

// CHECK-LABEL: test_vec_vec_muli
// CHECK: [[A:%.*]] = load <3 x i32>, ptr %a.addr
// CHECK: [[B:%.*]] = load <3 x i32>, ptr %b.addr
// DXIL: %hlsl.mul = call i32 @llvm.dx.sdot.v3i32(<3 x i32> [[A]], <3 x i32> [[B]])
// SPIRV: %hlsl.mul = call i32 @llvm.spv.sdot.v3i32(<3 x i32> [[A]], <3 x i32> [[B]])
// CHECK: ret i32 %hlsl.mul
export int test_vec_vec_muli(int3 a, int3 b) { return mul(a, b); }

// CHECK-LABEL: test_vec_vec_mulu
// CHECK: [[A:%.*]] = load <3 x i32>, ptr %a.addr
// CHECK: [[B:%.*]] = load <3 x i32>, ptr %b.addr
// DXIL: %hlsl.mul = call i32 @llvm.dx.udot.v3i32(<3 x i32> [[A]], <3 x i32> [[B]])
// SPIRV: %hlsl.mul = call i32 @llvm.spv.udot.v3i32(<3 x i32> [[A]], <3 x i32> [[B]])
// CHECK: ret i32 %hlsl.mul
export uint test_vec_vec_mulu(uint3 a, uint3 b) { return mul(a, b); }

// Double vector dot product: DXIL uses scalar arithmetic, SPIR-V uses fdot
// CHECK-LABEL: test_vec_vec_muld
// CHECK: [[A:%.*]] = load <3 x double>, ptr %a.addr
// CHECK: [[B:%.*]] = load <3 x double>, ptr %b.addr
// DXIL-NOT: @llvm.dx.fdot
// DXIL: [[A0:%.*]] = extractelement <3 x double> [[A]], i64 0
// DXIL: [[B0:%.*]] = extractelement <3 x double> [[B]], i64 0
// DXIL: [[MUL0:%.*]] = fmul {{.*}} double [[A0]], [[B0]]
// DXIL: [[A1:%.*]] = extractelement <3 x double> [[A]], i64 1
// DXIL: [[B1:%.*]] = extractelement <3 x double> [[B]], i64 1
// DXIL: [[FMA0:%.*]] = call {{.*}} double @llvm.fmuladd.f64(double [[A1]], double [[B1]], double [[MUL0]])
// DXIL: [[A2:%.*]] = extractelement <3 x double> [[A]], i64 2
// DXIL: [[B2:%.*]] = extractelement <3 x double> [[B]], i64 2
// DXIL: [[FMA1:%.*]] = call {{.*}} double @llvm.fmuladd.f64(double [[A2]], double [[B2]], double [[FMA0]])
// DXIL: ret double [[FMA1]]
// SPIRV: %hlsl.mul = call {{.*}} double @llvm.spv.fdot.v3f64(<3 x double> [[A]], <3 x double> [[B]])
// SPIRV: ret double %hlsl.mul
export double test_vec_vec_muld(double3 a, double3 b) { return mul(a, b); }

// -- Case 6: vector * matrix -> vector --

// CHECK-LABEL: test_vec_mat_mul
// CHECK: [[V:%.*]] = load <2 x float>, ptr %v.addr
// CHECK: [[M:%.*]] = load <6 x float>, ptr %m.addr
// CHECK: %hlsl.mul = call {{.*}} <3 x float> @llvm.matrix.multiply.v3f32.v2f32.v6f32(<2 x float> [[V]], <6 x float> [[M]], i32 1, i32 2, i32 3)
// CHECK: ret <3 x float> %hlsl.mul
export float3 test_vec_mat_mul(float2 v, float2x3 m) { return mul(v, m); }

// -- Case 7: matrix * scalar -> matrix --

// CHECK-LABEL: test_mat_scalar_mul
// CHECK: [[A:%.*]] = load <6 x float>, ptr %a.addr
// CHECK: [[B:%.*]] = load float, ptr %b.addr
// CHECK: %.splatinsert = insertelement <6 x float> poison, float [[B]], i64 0
// CHECK: %.splat = shufflevector <6 x float> %.splatinsert, <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: %hlsl.mul = fmul {{.*}} <6 x float> %.splat, [[A]]
// CHECK: ret <6 x float> %hlsl.mul
export float2x3 test_mat_scalar_mul(float2x3 a, float b) { return mul(a, b); }

// -- Case 8: matrix * vector -> vector --

// CHECK-LABEL: test_mat_vec_mul
// CHECK: [[M:%.*]] = load <6 x float>, ptr %m.addr
// CHECK: [[V:%.*]] = load <3 x float>, ptr %v.addr
// CHECK: %hlsl.mul = call {{.*}} <2 x float> @llvm.matrix.multiply.v2f32.v6f32.v3f32(<6 x float> [[M]], <3 x float> [[V]], i32 2, i32 3, i32 1)
// CHECK: ret <2 x float> %hlsl.mul
export float2 test_mat_vec_mul(float2x3 m, float3 v) { return mul(m, v); }

// -- Case 9: matrix * matrix -> matrix --

// CHECK-LABEL: test_mat_mat_mul
// CHECK: [[A:%.*]] = load <6 x float>, ptr %a.addr
// CHECK: [[B:%.*]] = load <12 x float>, ptr %b.addr
// CHECK: %hlsl.mul = call {{.*}} <8 x float> @llvm.matrix.multiply.v8f32.v6f32.v12f32(<6 x float> [[A]], <12 x float> [[B]], i32 2, i32 3, i32 4)
// CHECK: ret <8 x float> %hlsl.mul
export float2x4 test_mat_mat_mul(float2x3 a, float3x4 b) { return mul(a, b); }

// -- Integer matrix multiply --

// CHECK-LABEL: test_mat_mat_muli
// CHECK: [[A:%.*]] = load <6 x i32>, ptr %a.addr
// CHECK: [[B:%.*]] = load <12 x i32>, ptr %b.addr
// CHECK: %hlsl.mul = call <8 x i32> @llvm.matrix.multiply.v8i32.v6i32.v12i32(<6 x i32> [[A]], <12 x i32> [[B]], i32 2, i32 3, i32 4)
// CHECK: ret <8 x i32> %hlsl.mul
export int2x4 test_mat_mat_muli(int2x3 a, int3x4 b) { return mul(a, b); }
