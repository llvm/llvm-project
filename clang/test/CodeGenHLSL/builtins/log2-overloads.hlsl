// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_log2_double
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.log2.f32(
float test_log2_double(double p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_log2_double2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.log2.v2f32
float2 test_log2_double2(double2 p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_log2_double3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.log2.v3f32
float3 test_log2_double3(double3 p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_log2_double4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.log2.v4f32
float4 test_log2_double4(double4 p0) { return log2(p0); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_log2_int
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.log2.f32(
float test_log2_int(int p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_log2_int2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.log2.v2f32
float2 test_log2_int2(int2 p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_log2_int3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.log2.v3f32
float3 test_log2_int3(int3 p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_log2_int4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.log2.v4f32
float4 test_log2_int4(int4 p0) { return log2(p0); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_log2_uint
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.log2.f32(
float test_log2_uint(uint p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_log2_uint2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.log2.v2f32
float2 test_log2_uint2(uint2 p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_log2_uint3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.log2.v3f32
float3 test_log2_uint3(uint3 p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_log2_uint4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.log2.v4f32
float4 test_log2_uint4(uint4 p0) { return log2(p0); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_log2_int64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.log2.f32(
float test_log2_int64_t(int64_t p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_log2_int64_t2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.log2.v2f32
float2 test_log2_int64_t2(int64_t2 p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_log2_int64_t3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.log2.v3f32
float3 test_log2_int64_t3(int64_t3 p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_log2_int64_t4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.log2.v4f32
float4 test_log2_int64_t4(int64_t4 p0) { return log2(p0); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_log2_uint64_t
// CHECK: call reassoc nnan ninf nsz arcp afn float @llvm.log2.f32(
float test_log2_uint64_t(uint64_t p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_log2_uint64_t2
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.log2.v2f32
float2 test_log2_uint64_t2(uint64_t2 p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_log2_uint64_t3
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.log2.v3f32
float3 test_log2_uint64_t3(uint64_t3 p0) { return log2(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_log2_uint64_t4
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.log2.v4f32
float4 test_log2_uint64_t4(uint64_t4 p0) { return log2(p0); }
