// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s

// CHECK: define noundef i1 @
// CHECK: %dx.isinf = call i1 @llvm.dx.isinf.f32(
// CHECK: ret i1 %dx.isinf
bool test_isinf_double(double p0) { return isinf(p0); }
// CHECK: define noundef <2 x i1> @
// CHECK: %dx.isinf = call <2 x i1> @llvm.dx.isinf.v2f32
// CHECK: ret <2 x i1> %dx.isinf
bool2 test_isinf_double2(double2 p0) { return isinf(p0); }
// CHECK: define noundef <3 x i1> @
// CHECK: %dx.isinf = call <3 x i1> @llvm.dx.isinf.v3f32
// CHECK: ret <3 x i1> %dx.isinf
bool3 test_isinf_double3(double3 p0) { return isinf(p0); }
// CHECK: define noundef <4 x i1> @
// CHECK: %dx.isinf = call <4 x i1> @llvm.dx.isinf.v4f32
// CHECK: ret <4 x i1> %dx.isinf
bool4 test_isinf_double4(double4 p0) { return isinf(p0); }
