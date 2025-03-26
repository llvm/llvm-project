// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) half @_Z14test_sqrt_half
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn half @llvm.sqrt.f16(
// NATIVE_HALF: ret half %{{.*}}
// NO_HALF-LABEL: define noundef nofpclass(nan inf) float @_Z14test_sqrt_half
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(
// NO_HALF: ret float %{{.*}}
half test_sqrt_half(half p0) { return sqrt(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <2 x half> @_Z15test_sqrt_half2
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.sqrt.v2f16
// NATIVE_HALF: ret <2 x half> %{{.*}}
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z15test_sqrt_half2
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sqrt.v2f32(
// NO_HALF: ret <2 x float> %{{.*}}
half2 test_sqrt_half2(half2 p0) { return sqrt(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <3 x half> @_Z15test_sqrt_half3
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.sqrt.v3f16
// NATIVE_HALF: ret <3 x half> %{{.*}}
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z15test_sqrt_half3
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sqrt.v3f32(
// NO_HALF: ret <3 x float> %{{.*}}
half3 test_sqrt_half3(half3 p0) { return sqrt(p0); }
// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <4 x half> @_Z15test_sqrt_half4
// NATIVE_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.sqrt.v4f16
// NATIVE_HALF: ret <4 x half> %{{.*}}
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z15test_sqrt_half4
// NO_HALF: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sqrt.v4f32(
// NO_HALF: ret <4 x float> %{{.*}}
half4 test_sqrt_half4(half4 p0) { return sqrt(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float @_Z15test_sqrt_float
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(
// CHECK: ret float %{{.*}}
float test_sqrt_float(float p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> @_Z16test_sqrt_float2
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sqrt.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_sqrt_float2(float2 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> @_Z16test_sqrt_float3
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sqrt.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_sqrt_float3(float3 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> @_Z16test_sqrt_float4
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sqrt.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_sqrt_float4(float4 p0) { return sqrt(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_sqrt_double
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(
// CHECK: ret float %{{.*}}
float test_sqrt_double(double p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_sqrt_double2
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sqrt.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_sqrt_double2(double2 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_sqrt_double3
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sqrt.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_sqrt_double3(double3 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_sqrt_double4
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sqrt.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_sqrt_double4(double4 p0) { return sqrt(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_sqrt_int
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(
// CHECK: ret float %{{.*}}
float test_sqrt_int(int p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_sqrt_int2
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sqrt.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_sqrt_int2(int2 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_sqrt_int3
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sqrt.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_sqrt_int3(int3 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_sqrt_int4
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sqrt.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_sqrt_int4(int4 p0) { return sqrt(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_sqrt_uint
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(
// CHECK: ret float %{{.*}}
float test_sqrt_uint(uint p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_sqrt_uint2
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sqrt.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_sqrt_uint2(uint2 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_sqrt_uint3
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sqrt.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_sqrt_uint3(uint3 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_sqrt_uint4
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sqrt.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_sqrt_uint4(uint4 p0) { return sqrt(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_sqrt_int64_t
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(
// CHECK: ret float %{{.*}}
float test_sqrt_int64_t(int64_t p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_sqrt_int64_t2
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sqrt.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_sqrt_int64_t2(int64_t2 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_sqrt_int64_t3
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sqrt.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_sqrt_int64_t3(int64_t3 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_sqrt_int64_t4
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sqrt.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_sqrt_int64_t4(int64_t4 p0) { return sqrt(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_sqrt_uint64_t
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.sqrt.f32(
// CHECK: ret float %{{.*}}
float test_sqrt_uint64_t(uint64_t p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_sqrt_uint64_t2
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.sqrt.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_sqrt_uint64_t2(uint64_t2 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_sqrt_uint64_t3
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.sqrt.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_sqrt_uint64_t3(uint64_t3 p0) { return sqrt(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_sqrt_uint64_t4
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.sqrt.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_sqrt_uint64_t4(uint64_t4 p0) { return sqrt(p0); }
