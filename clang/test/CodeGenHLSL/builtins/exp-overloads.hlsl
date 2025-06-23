// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_exp_double
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(
// CHECK: ret float %elt.exp
float test_exp_double(double p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_exp_double2
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.exp.v2f32
// CHECK: ret <2 x float> %elt.exp
float2 test_exp_double2(double2 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_exp_double3
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.exp.v3f32
// CHECK: ret <3 x float> %elt.exp
float3 test_exp_double3(double3 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_exp_double4
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.exp.v4f32
// CHECK: ret <4 x float> %elt.exp
float4 test_exp_double4(double4 p0) { return exp(p0); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_exp_int
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(
// CHECK: ret float %elt.exp
float test_exp_int(int p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_exp_int2
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.exp.v2f32
// CHECK: ret <2 x float> %elt.exp
float2 test_exp_int2(int2 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_exp_int3
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.exp.v3f32
// CHECK: ret <3 x float> %elt.exp
float3 test_exp_int3(int3 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_exp_int4
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.exp.v4f32
// CHECK: ret <4 x float> %elt.exp
float4 test_exp_int4(int4 p0) { return exp(p0); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_exp_uint
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(
// CHECK: ret float %elt.exp
float test_exp_uint(uint p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_exp_uint2
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.exp.v2f32
// CHECK: ret <2 x float> %elt.exp
float2 test_exp_uint2(uint2 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_exp_uint3
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.exp.v3f32
// CHECK: ret <3 x float> %elt.exp
float3 test_exp_uint3(uint3 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_exp_uint4
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.exp.v4f32
// CHECK: ret <4 x float> %elt.exp
float4 test_exp_uint4(uint4 p0) { return exp(p0); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_exp_int64_t
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(
// CHECK: ret float %elt.exp
float test_exp_int64_t(int64_t p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_exp_int64_t2
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.exp.v2f32
// CHECK: ret <2 x float> %elt.exp
float2 test_exp_int64_t2(int64_t2 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_exp_int64_t3
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.exp.v3f32
// CHECK: ret <3 x float> %elt.exp
float3 test_exp_int64_t3(int64_t3 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_exp_int64_t4
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.exp.v4f32
// CHECK: ret <4 x float> %elt.exp
float4 test_exp_int64_t4(int64_t4 p0) { return exp(p0); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_exp_uint64_t
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn float @llvm.exp.f32(
// CHECK: ret float %elt.exp
float test_exp_uint64_t(uint64_t p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_exp_uint64_t2
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.exp.v2f32
// CHECK: ret <2 x float> %elt.exp
float2 test_exp_uint64_t2(uint64_t2 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_exp_uint64_t3
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.exp.v3f32
// CHECK: ret <3 x float> %elt.exp
float3 test_exp_uint64_t3(uint64_t3 p0) { return exp(p0); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_exp_uint64_t4
// CHECK: %elt.exp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.exp.v4f32
// CHECK: ret <4 x float> %elt.exp
float4 test_exp_uint64_t4(uint64_t4 p0) { return exp(p0); }
