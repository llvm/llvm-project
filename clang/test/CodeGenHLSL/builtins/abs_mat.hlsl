// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// CHECK-LABEL: test_abs_int2x3
// CHECK: call <6 x i32> @llvm.abs.v6i32(<6 x i32> %{{.*}}, i1 false)
int2x3 test_abs_int2x3(int2x3 p0) { return abs(p0); }

// CHECK-LABEL: test_abs_int3x4
// CHECK: call <12 x i32> @llvm.abs.v12i32(<12 x i32> %{{.*}}, i1 false)
int3x4 test_abs_int3x4(int3x4 p0) { return abs(p0); }

// CHECK-LABEL: test_abs_int4x4
// CHECK: call <16 x i32> @llvm.abs.v16i32(<16 x i32> %{{.*}}, i1 false)
int4x4 test_abs_int4x4(int4x4 p0) { return abs(p0); }

// CHECK-LABEL: test_abs_uint2x3
// CHECK: call {{.*}} @{{.*}}hlsl3abs
uint2x3 test_abs_uint2x3(uint2x3 p0) { return abs(p0); }

// CHECK-LABEL: test_abs_uint3x4
// CHECK: call {{.*}} @{{.*}}hlsl3abs
uint3x4 test_abs_uint3x4(uint3x4 p0) { return abs(p0); }

// CHECK-LABEL: test_abs_uint4x4
// CHECK: call {{.*}} @{{.*}}hlsl3abs
uint4x4 test_abs_uint4x4(uint4x4 p0) { return abs(p0); }

// CHECK-LABEL: test_abs_half2x3
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <6 x half> @llvm.fabs.v6f16(<6 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <6 x float> @llvm.fabs.v6f32(<6 x float> %{{.*}})
half2x3 test_abs_half2x3(half2x3 p0) { return abs(p0); }

// CHECK-LABEL: test_abs_half3x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <12 x half> @llvm.fabs.v12f16(<12 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <12 x float> @llvm.fabs.v12f32(<12 x float> %{{.*}})
half3x4 test_abs_half3x4(half3x4 p0) { return abs(p0); }

// CHECK-LABEL: test_abs_half4x4
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <16 x half> @llvm.fabs.v16f16(<16 x half> %{{.*}})
// NO_HALF: call reassoc nnan ninf nsz arcp afn <16 x float> @llvm.fabs.v16f32(<16 x float> %{{.*}})
half4x4 test_abs_half4x4(half4x4 p0) { return abs(p0); }
