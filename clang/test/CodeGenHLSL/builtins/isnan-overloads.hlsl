// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s

// CHECK: define hidden noundef i1 @
// CHECK: %hlsl.isnan = call i1 @llvm.dx.isnan.f32(
// CHECK: ret i1 %hlsl.isnan
bool test_isnan_double(double p0) { return isnan(p0); }
// CHECK: define hidden noundef <2 x i1> @
// CHECK: %hlsl.isnan = call <2 x i1> @llvm.dx.isnan.v2f32
// CHECK: ret <2 x i1> %hlsl.isnan
bool2 test_isnan_double2(double2 p0) { return isnan(p0); }
// CHECK: define hidden noundef <3 x i1> @
// CHECK: %hlsl.isnan = call <3 x i1> @llvm.dx.isnan.v3f32
// CHECK: ret <3 x i1> %hlsl.isnan
bool3 test_isnan_double3(double3 p0) { return isnan(p0); }
// CHECK: define hidden noundef <4 x i1> @
// CHECK: %hlsl.isnan = call <4 x i1> @llvm.dx.isnan.v4f32
// CHECK: ret <4 x i1> %hlsl.isnan
bool4 test_isnan_double4(double4 p0) { return isnan(p0); }
