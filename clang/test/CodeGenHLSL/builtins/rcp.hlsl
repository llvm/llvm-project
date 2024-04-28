// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,DXIL_CHECK,DXIL_NATIVE_HALF,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,DXIL_CHECK,NO_HALF,DXIL_NO_HALF
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \ 
// RUN:   --check-prefixes=CHECK,NATIVE_HALF,SPIR_NATIVE_HALF,SPIR_CHECK
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF,SPIR_NO_HALF,SPIR_CHECK

// DXIL_NATIVE_HALF: define noundef half @
// SPIR_NATIVE_HALF: define spir_func noundef half @
// NATIVE_HALF: %hlsl.rcp = fdiv half 0xH3C00, %{{.*}} 
// NATIVE_HALF: ret half %hlsl.rcp
// DXIL_NO_HALF: define noundef float @
// SPIR_NO_HALF: define spir_func noundef float @
// NO_HALF: %hlsl.rcp = fdiv float 1.000000e+00, %{{.*}}
// NO_HALF: ret float %hlsl.rcp
half test_rcp_half(half p0) { return rcp(p0); }

// DXIL_NATIVE_HALF: define noundef <2 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <2 x half> @
// NATIVE_HALF: %hlsl.rcp = fdiv <2 x half> <half  0xH3C00, half  0xH3C00>, %{{.*}} 
// NATIVE_HALF: ret <2 x half> %hlsl.rcp
// DXIL_NO_HALF: define noundef <2 x float> @
// SPIR_NO_HALF: define spir_func noundef <2 x float> @
// NO_HALF: %hlsl.rcp = fdiv <2 x float> <float 1.000000e+00, float 1.000000e+00>, %{{.*}}
// NO_HALF: ret <2 x float> %hlsl.rcp
half2 test_rcp_half2(half2 p0) { return rcp(p0); }

// DXIL_NATIVE_HALF: define noundef <3 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <3 x half> @
// NATIVE_HALF: %hlsl.rcp = fdiv <3 x half> <half  0xH3C00, half  0xH3C00, half  0xH3C00>, %{{.*}} 
// NATIVE_HALF: ret <3 x half> %hlsl.rcp
// DXIL_NO_HALF: define noundef <3 x float> @
// SPIR_NO_HALF: define spir_func noundef <3 x float> @
// NO_HALF: %hlsl.rcp = fdiv <3 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %{{.*}}
// NO_HALF: ret <3 x float> %hlsl.rcp
half3 test_rcp_half3(half3 p0) { return rcp(p0); }

// DXIL_NATIVE_HALF: define noundef <4 x half> @
// SPIR_NATIVE_HALF: define spir_func noundef <4 x half> @
// NATIVE_HALF: %hlsl.rcp = fdiv <4 x half> <half  0xH3C00, half  0xH3C00, half  0xH3C00, half  0xH3C00>, %{{.*}} 
// NATIVE_HALF: ret <4 x half> %hlsl.rcp
// DXIL_NO_HALF: define noundef <4 x float> @
// SPIR_NO_HALF: define spir_func noundef <4 x float> @
// NO_HALF: %hlsl.rcp = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %{{.*}}
// NO_HALF: ret <4 x float> %hlsl.rcp
half4 test_rcp_half4(half4 p0) { return rcp(p0); }

// DXIL_CHECK: define noundef float @
// SPIR_CHECK: define spir_func noundef float @
// CHECK: %hlsl.rcp = fdiv float 1.000000e+00, %{{.*}}
// CHECK: ret float %hlsl.rcp
float test_rcp_float(float p0) { return rcp(p0); }

// DXIL_CHECK: define noundef <2 x float> @
// SPIR_CHECK: define spir_func noundef <2 x float> @
// CHECK: %hlsl.rcp = fdiv <2 x float> <float 1.000000e+00, float 1.000000e+00>, %{{.*}}
// CHECK: ret <2 x float> %hlsl.rcp
float2 test_rcp_float2(float2 p0) { return rcp(p0); }

// DXIL_CHECK: define noundef <3 x float> @
// SPIR_CHECK: define spir_func noundef <3 x float> @
// CHECK: %hlsl.rcp = fdiv <3 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %{{.*}}
// CHECK: ret <3 x float> %hlsl.rcp
float3 test_rcp_float3(float3 p0) { return rcp(p0); }

// DXIL_CHECK: define noundef <4 x float> @
// SPIR_CHECK: define spir_func noundef <4 x float> @
// CHECK: %hlsl.rcp = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %{{.*}}
// CHECK: ret <4 x float> %hlsl.rcp
float4 test_rcp_float4(float4 p0) { return rcp(p0); }

// DXIL_CHECK: define noundef double @
// SPIR_CHECK: define spir_func noundef double @
// CHECK: %hlsl.rcp = fdiv double 1.000000e+00, %{{.*}} 
// CHECK: ret double %hlsl.rcp
double test_rcp_double(double p0) { return rcp(p0); }

// DXIL_CHECK: define noundef <2 x double> @
// SPIR_CHECK: define spir_func noundef <2 x double> @
// CHECK: %hlsl.rcp = fdiv <2 x double> <double 1.000000e+00, double 1.000000e+00>, %{{.*}}
// CHECK: ret <2 x double> %hlsl.rcp
double2 test_rcp_double2(double2 p0) { return rcp(p0); }

// DXIL_CHECK: define noundef <3 x double> @
// SPIR_CHECK: define spir_func noundef <3 x double> @
// CHECK: %hlsl.rcp = fdiv <3 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>, %{{.*}}
// CHECK: ret <3 x double> %hlsl.rcp
double3 test_rcp_double3(double3 p0) { return rcp(p0); }

// DXIL_CHECK: define noundef <4 x double> @
// SPIR_CHECK: define spir_func noundef <4 x double> @
// CHECK: %hlsl.rcp = fdiv <4 x double> <double 1.000000e+00, double 1.000000e+00, double 1.000000e+00, double 1.000000e+00>, %{{.*}}
// CHECK: ret <4 x double> %hlsl.rcp
double4 test_rcp_double4(double4 p0) { return rcp(p0); }
