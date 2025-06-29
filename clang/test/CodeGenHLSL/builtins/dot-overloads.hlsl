// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:  -DTARGET=dx -DFNATTRS=noundef -DFFNATTRS="nofpclass(nan inf)"

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple spirv-unknown-vulkan-compute %s \
// RUN:  -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:  -DTARGET=spv -DFNATTRS="spir_func noundef" -DFFNATTRS="nofpclass(nan inf)"

// CHECK: define [[FNATTRS]] [[FFNATTRS]] float {{.*}}test_dot_float4_mismatch1
// CHECK: [[CONV0:%.*]] = insertelement <4 x float> poison, float %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x float> [[CONV0]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: [[DOT:%.*]] = call {{.*}} float @llvm.[[TARGET]].fdot.v4f32(<4 x float> %{{.*}}, <4 x float> [[CONV1]])
// CHECK: ret float [[DOT]]
float test_dot_float4_mismatch1(float4 p0, float p1) { return dot(p0, p1); }

// CHECK: define [[FNATTRS]] [[FFNATTRS]] float {{.*}}test_dot_float4_mismatch2
// CHECK: [[CONV0:%.*]] = insertelement <4 x float> poison, float %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x float> [[CONV0]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: [[DOT:%.*]] = call {{.*}} float @llvm.[[TARGET]].fdot.v4f32(<4 x float> [[CONV1]], <4 x float> %{{.*}})
// CHECK: ret float [[DOT]]
float test_dot_float4_mismatch2(float4 p0, float p1) { return dot(p1, p0); }

// CHECK: define [[FNATTRS]] i32 {{.*}}test_dot_int2_mismatch1
// CHECK: [[CONV0:%.*]] = insertelement <2 x i32> poison, i32 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <2 x i32> [[CONV0]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK: [[DOT:%.*]] = call {{.*}} i32 @llvm.[[TARGET]].sdot.v2i32(<2 x i32> %{{.*}}, <2 x i32> [[CONV1]])
// CHECK: ret i32 [[DOT]]
int test_dot_int2_mismatch1(int2 p0, int p1) { return dot(p0, p1); }

