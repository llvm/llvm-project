// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_round_double
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn float @llvm.roundeven.f32(
// CHECK: ret float %elt.roundeven
float test_round_double(double p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_round_double2
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.roundeven.v2f32
// CHECK: ret <2 x float> %elt.roundeven
float2 test_round_double2(double2 p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_round_double3
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.roundeven.v3f32
// CHECK: ret <3 x float> %elt.roundeven
float3 test_round_double3(double3 p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_round_double4
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.roundeven.v4f32
// CHECK: ret <4 x float> %elt.roundeven
float4 test_round_double4(double4 p0) { return round(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_round_int
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn float @llvm.roundeven.f32(
// CHECK: ret float %elt.roundeven
float test_round_int(int p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_round_int2
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.roundeven.v2f32
// CHECK: ret <2 x float> %elt.roundeven
float2 test_round_int2(int2 p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_round_int3
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.roundeven.v3f32
// CHECK: ret <3 x float> %elt.roundeven
float3 test_round_int3(int3 p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_round_int4
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.roundeven.v4f32
// CHECK: ret <4 x float> %elt.roundeven
float4 test_round_int4(int4 p0) { return round(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_round_uint
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn float @llvm.roundeven.f32(
// CHECK: ret float %elt.roundeven
float test_round_uint(uint p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_round_uint2
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.roundeven.v2f32
// CHECK: ret <2 x float> %elt.roundeven
float2 test_round_uint2(uint2 p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_round_uint3
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.roundeven.v3f32
// CHECK: ret <3 x float> %elt.roundeven
float3 test_round_uint3(uint3 p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_round_uint4
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.roundeven.v4f32
// CHECK: ret <4 x float> %elt.roundeven
float4 test_round_uint4(uint4 p0) { return round(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_round_int64_t
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn float @llvm.roundeven.f32(
// CHECK: ret float %elt.roundeven
float test_round_int64_t(int64_t p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_round_int64_t2
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.roundeven.v2f32
// CHECK: ret <2 x float> %elt.roundeven
float2 test_round_int64_t2(int64_t2 p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_round_int64_t3
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.roundeven.v3f32
// CHECK: ret <3 x float> %elt.roundeven
float3 test_round_int64_t3(int64_t3 p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_round_int64_t4
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.roundeven.v4f32
// CHECK: ret <4 x float> %elt.roundeven
float4 test_round_int64_t4(int64_t4 p0) { return round(p0); }

// CHECK-LABEL: define noundef nofpclass(nan inf) float {{.*}}test_round_uint64_t
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn float @llvm.roundeven.f32(
// CHECK: ret float %elt.roundeven
float test_round_uint64_t(uint64_t p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <2 x float> {{.*}}test_round_uint64_t2
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.roundeven.v2f32
// CHECK: ret <2 x float> %elt.roundeven
float2 test_round_uint64_t2(uint64_t2 p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <3 x float> {{.*}}test_round_uint64_t3
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.roundeven.v3f32
// CHECK: ret <3 x float> %elt.roundeven
float3 test_round_uint64_t3(uint64_t3 p0) { return round(p0); }
// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_round_uint64_t4
// CHECK: %elt.roundeven = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.roundeven.v4f32
// CHECK: ret <4 x float> %elt.roundeven
float4 test_round_uint64_t4(uint64_t4 p0) { return round(p0); }
