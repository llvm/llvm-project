// RUN: %clang_cc1 -finclude-default-header -O1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -finclude-default-header -O1 -triple spirv-unknown-vulkan1.3-library -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,SPIRV

// -- Case 1: scalar * scalar -> scalar --

// CHECK-LABEL: test_scalar_mulf
// CHECK: %mul.i = fmul {{.*}} float %b, %a
// CHECK: ret float %mul.i
export float test_scalar_mulf(float a, float b) { return mul(a, b); }

// CHECK-LABEL: test_scalar_muli
// CHECK: %mul.i = mul {{.*}} i32 %b, %a
// CHECK: ret i32 %mul.i
export int test_scalar_muli(int a, int b) { return mul(a, b); }

// -- Case 2: scalar * vector -> vector --

// CHECK-LABEL: test_scalar_vec_mul
// CHECK: %splat.splatinsert.i = insertelement <3 x float> poison, float %a, i64 0
// CHECK: %splat.splat.i = shufflevector <3 x float> %splat.splatinsert.i, <3 x float> poison, <3 x i32> zeroinitializer
// CHECK: %mul.i = fmul {{.*}} <3 x float> %splat.splat.i, %b
// CHECK: ret <3 x float> %mul.i
export float3 test_scalar_vec_mul(float a, float3 b) { return mul(a, b); }

// -- Case 3: scalar * matrix -> matrix --

// CHECK-LABEL: test_scalar_mat_mul
// CHECK: %scalar.splat.splatinsert.i = insertelement <6 x float> poison, float %a, i64 0
// CHECK: %scalar.splat.splat.i = shufflevector <6 x float> %scalar.splat.splatinsert.i, <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: [[MUL:%.*]] = fmul {{.*}} <6 x float> %scalar.splat.splat.i, %b
// CHECK: ret <6 x float> [[MUL]]
export float2x3 test_scalar_mat_mul(float a, float2x3 b) { return mul(a, b); }

// -- Case 4: vector * scalar -> vector --

// CHECK-LABEL: test_vec_scalar_mul
// CHECK: %splat.splatinsert.i = insertelement <3 x float> poison, float %b, i64 0
// CHECK: %splat.splat.i = shufflevector <3 x float> %splat.splatinsert.i, <3 x float> poison, <3 x i32> zeroinitializer
// CHECK: %mul.i = fmul {{.*}} <3 x float> %splat.splat.i, %a
// CHECK: ret <3 x float> %mul.i
export float3 test_vec_scalar_mul(float3 a, float b) { return mul(a, b); }

// -- Case 5: vector * vector -> scalar (dot product) --

// CHECK-LABEL: test_vec_vec_mul
// DXIL: %hlsl.dot.i = {{.*}} call {{.*}} float @llvm.dx.fdot.v3f32(<3 x float> {{.*}} %a, <3 x float> {{.*}} %b)
// SPIRV: %hlsl.dot.i = {{.*}} call {{.*}} float @llvm.spv.fdot.v3f32(<3 x float> {{.*}} %a, <3 x float> {{.*}} %b)
// CHECK: ret float %hlsl.dot.i
export float test_vec_vec_mul(float3 a, float3 b) { return mul(a, b); }

// CHECK-LABEL: test_vec_vec_muli
// DXIL: %hlsl.dot.i = {{.*}} call {{.*}} i32 @llvm.dx.sdot.v3i32(<3 x i32> %a, <3 x i32> %b)
// SPIRV: %hlsl.dot.i = {{.*}} call {{.*}} i32 @llvm.spv.sdot.v3i32(<3 x i32> %a, <3 x i32> %b)
// CHECK: ret i32 %hlsl.dot.i
export int test_vec_vec_muli(int3 a, int3 b) { return mul(a, b); }

// CHECK-LABEL: test_vec_vec_mulu
// DXIL: %hlsl.dot.i = {{.*}} call {{.*}} i32 @llvm.dx.udot.v3i32(<3 x i32> %a, <3 x i32> %b)
// SPIRV: %hlsl.dot.i = {{.*}} call {{.*}} i32 @llvm.spv.udot.v3i32(<3 x i32> %a, <3 x i32> %b)
// CHECK: ret i32 %hlsl.dot.i
export uint test_vec_vec_mulu(uint3 a, uint3 b) { return mul(a, b); }

// Double vector dot product: no dot intrinsic for double vectors.
// The checks for this test are less precise because the scalar loop may be vectorized depending on the build configuration.
// CHECK-LABEL: test_vec_vec_muld
// CHECK-NOT: @llvm.dx.fdot
// CHECK-NOT: @llvm.spv.fdot
// CHECK: fmul {{.*}} double
// CHECK: fadd {{.*}} double
// CHECK: ret double %{{.*}}
export double test_vec_vec_muld(double3 a, double3 b) { return mul(a, b); }

// -- Case 6: vector * matrix -> vector --

// CHECK-LABEL: test_vec_mat_mul
// CHECK: %hlsl.mul = {{.*}} call {{.*}} <3 x float> @llvm.matrix.multiply.v3f32.v2f32.v6f32(<2 x float> %v, <6 x float> %m, i32 1, i32 2, i32 3)
// CHECK: ret <3 x float> %hlsl.mul
export float3 test_vec_mat_mul(float2 v, float2x3 m) { return mul(v, m); }

// -- Case 7: matrix * scalar -> matrix --

// CHECK-LABEL: test_mat_scalar_mul
// CHECK: %scalar.splat.splatinsert.i = insertelement <6 x float> poison, float %b, i64 0
// CHECK: %scalar.splat.splat.i = shufflevector <6 x float> %scalar.splat.splatinsert.i, <6 x float> poison, <6 x i32> zeroinitializer
// CHECK: [[MUL:%.*]] = fmul {{.*}} <6 x float> %scalar.splat.splat.i, %a
// CHECK: ret <6 x float> [[MUL]]
export float2x3 test_mat_scalar_mul(float2x3 a, float b) { return mul(a, b); }

// -- Case 8: matrix * vector -> vector --

// CHECK-LABEL: test_mat_vec_mul
// CHECK: %hlsl.mul = {{.*}} call {{.*}} <2 x float> @llvm.matrix.multiply.v2f32.v6f32.v3f32(<6 x float> %m, <3 x float> %v, i32 2, i32 3, i32 1)
// CHECK: ret <2 x float> %hlsl.mul
export float2 test_mat_vec_mul(float2x3 m, float3 v) { return mul(m, v); }

// -- Case 9: matrix * matrix -> matrix --

// CHECK-LABEL: test_mat_mat_mul
// CHECK: %hlsl.mul = {{.*}} call {{.*}} <8 x float> @llvm.matrix.multiply.v8f32.v6f32.v12f32(<6 x float> %a, <12 x float> %b, i32 2, i32 3, i32 4)
// CHECK: ret <8 x float> %hlsl.mul
export float2x4 test_mat_mat_mul(float2x3 a, float3x4 b) { return mul(a, b); }

// -- Integer matrix multiply --

// CHECK-LABEL: test_mat_mat_muli
// CHECK: %hlsl.mul = {{.*}} call <8 x i32> @llvm.matrix.multiply.v8i32.v6i32.v12i32(<6 x i32> %a, <12 x i32> %b, i32 2, i32 3, i32 4)
// CHECK: ret <8 x i32> %hlsl.mul
export int2x4 test_mat_mat_muli(int2x3 a, int3x4 b) { return mul(a, b); }
