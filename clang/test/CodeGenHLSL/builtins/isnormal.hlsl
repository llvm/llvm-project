// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.8-library -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=CHECK,HALF
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-unknown-vulkan-library -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=CHECK,HALF
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.9-library \
// RUN:   -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s --check-prefixes=CHECK,PROMOTED

// CHECK-LABEL: define {{.*}}test_float{{.*}}(
// CHECK: call i1 @llvm.is.fpclass.f32(float {{.*}}, {{.*}}i32 264)
bool test_float(float x) { return isnormal(x); }

// CHECK-LABEL: define {{.*}}test_half{{.*}}(
// HALF: call i1 @llvm.is.fpclass.f16(half {{.*}}, {{.*}}i32 264)
// PROMOTED: call i1 @llvm.is.fpclass.f32(float {{.*}}, {{.*}}i32 264)
bool test_half(half x) { return isnormal(x); }

// CHECK-LABEL: define {{.*}}test_vector{{.*}}(
// CHECK: call <3 x i1> @llvm.is.fpclass.v3f32(<3 x float> {{.*}}, {{.*}}i32 264)
bool3 test_vector(float3 x) { return isnormal(x); }

// CHECK-LABEL: define {{.*}}test_half_vector{{.*}}(
// HALF: call <2 x i1> @llvm.is.fpclass.v2f16(<2 x half> {{.*}}, {{.*}}i32 264)
// PROMOTED: call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> {{.*}}, {{.*}}i32 264)
bool2 test_half_vector(half2 x) { return isnormal(x); }

// CHECK-LABEL: define {{.*}}test_matrix{{.*}}(
// CHECK: call <6 x i1> @llvm.is.fpclass.v6f32(<6 x float> {{.*}}, {{.*}}i32 264)
bool2x3 test_matrix(float2x3 x) { return isnormal(x); }
