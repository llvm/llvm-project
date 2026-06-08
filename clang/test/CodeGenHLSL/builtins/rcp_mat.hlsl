// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

// CHECK-LABEL: test_rcp_half1x2
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <2 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <2 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <2 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <2 x float>
half1x2 test_rcp_half1x2(half1x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half1x3
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <3 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <3 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <3 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <3 x float>
half1x3 test_rcp_half1x3(half1x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half1x4
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <4 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <4 x float>
half1x4 test_rcp_half1x4(half1x4 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half2x1
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <2 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <2 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <2 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <2 x float>
half2x1 test_rcp_half2x1(half2x1 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half2x2
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <4 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <4 x float>
half2x2 test_rcp_half2x2(half2x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half2x3
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <6 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <6 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <6 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <6 x float>
half2x3 test_rcp_half2x3(half2x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half2x4
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <8 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <8 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <8 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <8 x float>
half2x4 test_rcp_half2x4(half2x4 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half3x1
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <3 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <3 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <3 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <3 x float>
half3x1 test_rcp_half3x1(half3x1 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half3x2
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <6 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <6 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <6 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <6 x float>
half3x2 test_rcp_half3x2(half3x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half3x3
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <9 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <9 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <9 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <9 x float>
half3x3 test_rcp_half3x3(half3x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half3x4
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <12 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <12 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <12 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <12 x float>
half3x4 test_rcp_half3x4(half3x4 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half4x1
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <4 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <4 x float>
half4x1 test_rcp_half4x1(half4x1 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half4x2
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <8 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <8 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <8 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <8 x float>
half4x2 test_rcp_half4x2(half4x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half4x3
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <12 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <12 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <12 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <12 x float>
half4x3 test_rcp_half4x3(half4x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_half4x4
// NATIVE_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <16 x half> splat (half 1.000000e+00), %{{.*}}
// NATIVE_HALF: ret <16 x half>
// NO_HALF: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <16 x float> splat (float 1.000000e+00), %{{.*}}
// NO_HALF: ret <16 x float>
half4x4 test_rcp_half4x4(half4x4 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float1x2
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <2 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <2 x float>
float1x2 test_rcp_float1x2(float1x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float1x3
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <3 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <3 x float>
float1x3 test_rcp_float1x3(float1x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float1x4
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <4 x float>
float1x4 test_rcp_float1x4(float1x4 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float2x1
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <2 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <2 x float>
float2x1 test_rcp_float2x1(float2x1 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float2x2
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <4 x float>
float2x2 test_rcp_float2x2(float2x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float2x3
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <6 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <6 x float>
float2x3 test_rcp_float2x3(float2x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float2x4
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <8 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <8 x float>
float2x4 test_rcp_float2x4(float2x4 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float3x1
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <3 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <3 x float>
float3x1 test_rcp_float3x1(float3x1 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float3x2
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <6 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <6 x float>
float3x2 test_rcp_float3x2(float3x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float3x3
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <9 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <9 x float>
float3x3 test_rcp_float3x3(float3x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float3x4
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <12 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <12 x float>
float3x4 test_rcp_float3x4(float3x4 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float4x1
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <4 x float>
float4x1 test_rcp_float4x1(float4x1 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float4x2
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <8 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <8 x float>
float4x2 test_rcp_float4x2(float4x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float4x3
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <12 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <12 x float>
float4x3 test_rcp_float4x3(float4x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_float4x4
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <16 x float> splat (float 1.000000e+00), %{{.*}}
// CHECK: ret <16 x float>
float4x4 test_rcp_float4x4(float4x4 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double1x2
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <2 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <2 x double>
double1x2 test_rcp_double1x2(double1x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double1x3
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <3 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <3 x double>
double1x3 test_rcp_double1x3(double1x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double1x4
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <4 x double>
double1x4 test_rcp_double1x4(double1x4 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double2x1
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <2 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <2 x double>
double2x1 test_rcp_double2x1(double2x1 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double2x2
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <4 x double>
double2x2 test_rcp_double2x2(double2x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double2x3
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <6 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <6 x double>
double2x3 test_rcp_double2x3(double2x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double2x4
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <8 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <8 x double>
double2x4 test_rcp_double2x4(double2x4 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double3x1
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <3 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <3 x double>
double3x1 test_rcp_double3x1(double3x1 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double3x2
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <6 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <6 x double>
double3x2 test_rcp_double3x2(double3x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double3x3
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <9 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <9 x double>
double3x3 test_rcp_double3x3(double3x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double3x4
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <12 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <12 x double>
double3x4 test_rcp_double3x4(double3x4 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double4x1
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <4 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <4 x double>
double4x1 test_rcp_double4x1(double4x1 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double4x2
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <8 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <8 x double>
double4x2 test_rcp_double4x2(double4x2 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double4x3
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <12 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <12 x double>
double4x3 test_rcp_double4x3(double4x3 p0) { return rcp(p0); }

// CHECK-LABEL: test_rcp_double4x4
// CHECK: %hlsl.rcp = fdiv reassoc nnan ninf nsz arcp afn <16 x double> splat (double 1.000000e+00), %{{.*}}
// CHECK: ret <16 x double>
double4x4 test_rcp_double4x4(double4x4 p0) { return rcp(p0); }

