// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:  -DFLOATATTRS="reassoc nnan ninf nsz arcp afn"

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_pow_double
// CHECK: [[CONV0:%.*]] = fptrunc [[FLOATATTRS]] double %{{.*}} to float
// CHECK: [[CONV1:%.*]] = fptrunc [[FLOATATTRS]] double %{{.*}} to float
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef float @llvm.pow.f32(float [[CONV0]], float [[CONV1]])
// CHECK: ret float [[POW]]
float test_pow_double(double p0, double p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_pow_double2
// CHECK: [[CONV0:%.*]] = fptrunc [[FLOATATTRS]] <2 x double> %{{.*}} to <2 x float>
// CHECK: [[CONV1:%.*]] = fptrunc [[FLOATATTRS]] <2 x double> %{{.*}} to <2 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <2 x float> @llvm.pow.v2f32(<2 x float> [[CONV0]], <2 x float> [[CONV1]])
// CHECK: ret <2 x float> [[POW]]
float2 test_pow_double2(double2 p0, double2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_pow_double3
// CHECK: [[CONV0:%.*]] = fptrunc [[FLOATATTRS]] <3 x double> %{{.*}} to <3 x float>
// CHECK: [[CONV1:%.*]] = fptrunc [[FLOATATTRS]] <3 x double> %{{.*}} to <3 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <3 x float> @llvm.pow.v3f32(<3 x float> [[CONV0]], <3 x float> [[CONV1]])
// CHECK: ret <3 x float> [[POW]]
float3 test_pow_double3(double3 p0, double3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_pow_double4
// CHECK: [[CONV0:%.*]] = fptrunc [[FLOATATTRS]] <4 x double> %{{.*}} to <4 x float>
// CHECK: [[CONV1:%.*]] = fptrunc [[FLOATATTRS]] <4 x double> %{{.*}} to <4 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <4 x float> @llvm.pow.v4f32(<4 x float> [[CONV0]], <4 x float> [[CONV1]])
// CHECK: ret <4 x float> [[POW]]
float4 test_pow_double4(double4 p0, double4 p1) { return pow(p0, p1); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_pow_int
// CHECK: [[CONV0:%.*]] = sitofp i32 %{{.*}} to float
// CHECK: [[CONV1:%.*]] = sitofp i32 %{{.*}} to float
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef float @llvm.pow.f32(float [[CONV0]], float [[CONV1]])
// CHECK: ret float [[POW]]
float test_pow_int(int p0, int p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_pow_int2
// CHECK: [[CONV0:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK: [[CONV1:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <2 x float> @llvm.pow.v2f32(<2 x float> [[CONV0]], <2 x float> [[CONV1]])
// CHECK: ret <2 x float> [[POW]]
float2 test_pow_int2(int2 p0, int2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_pow_int3
// CHECK: [[CONV0:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK: [[CONV1:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <3 x float> @llvm.pow.v3f32(<3 x float> [[CONV0]], <3 x float> [[CONV1]])
// CHECK: ret <3 x float> [[POW]]
float3 test_pow_int3(int3 p0, int3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_pow_int4
// CHECK: [[CONV0:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK: [[CONV1:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <4 x float> @llvm.pow.v4f32(<4 x float> [[CONV0]], <4 x float> [[CONV1]])
// CHECK: ret <4 x float> [[POW]]
float4 test_pow_int4(int4 p0, int4 p1) { return pow(p0, p1); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_pow_uint
// CHECK: [[CONV0:%.*]] = uitofp i32 %{{.*}} to float
// CHECK: [[CONV1:%.*]] = uitofp i32 %{{.*}} to float
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef float @llvm.pow.f32(float [[CONV0]], float [[CONV1]])
// CHECK: ret float [[POW]]
float test_pow_uint(uint p0, uint p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_pow_uint2
// CHECK: [[CONV0:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK: [[CONV1:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <2 x float> @llvm.pow.v2f32(<2 x float> [[CONV0]], <2 x float> [[CONV1]])
// CHECK: ret <2 x float> [[POW]]
float2 test_pow_uint2(uint2 p0, uint2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_pow_uint3
// CHECK: [[CONV0:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK: [[CONV1:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <3 x float> @llvm.pow.v3f32(<3 x float> [[CONV0]], <3 x float> [[CONV1]])
// CHECK: ret <3 x float> [[POW]]
float3 test_pow_uint3(uint3 p0, uint3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_pow_uint4
// CHECK: [[CONV0:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK: [[CONV1:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <4 x float> @llvm.pow.v4f32(<4 x float> [[CONV0]], <4 x float> [[CONV1]])
// CHECK: ret <4 x float> [[POW]]
float4 test_pow_uint4(uint4 p0, uint4 p1) { return pow(p0, p1); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_pow_int64_t
// CHECK: [[CONV0:%.*]] = sitofp i64 %{{.*}} to float
// CHECK: [[CONV1:%.*]] = sitofp i64 %{{.*}} to float
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef float @llvm.pow.f32(float [[CONV0]], float [[CONV1]])
// CHECK: ret float [[POW]]
float test_pow_int64_t(int64_t p0, int64_t p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_pow_int64_t2
// CHECK: [[CONV0:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK: [[CONV1:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <2 x float> @llvm.pow.v2f32(<2 x float> [[CONV0]], <2 x float> [[CONV1]])
// CHECK: ret <2 x float> [[POW]]
float2 test_pow_int64_t2(int64_t2 p0, int64_t2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_pow_int64_t3
// CHECK: [[CONV0:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK: [[CONV1:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <3 x float> @llvm.pow.v3f32(<3 x float> [[CONV0]], <3 x float> [[CONV1]])
// CHECK: ret <3 x float> [[POW]]
float3 test_pow_int64_t3(int64_t3 p0, int64_t3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_pow_int64_t4
// CHECK: [[CONV0:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK: [[CONV1:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <4 x float> @llvm.pow.v4f32(<4 x float> [[CONV0]], <4 x float> [[CONV1]])
// CHECK: ret <4 x float> [[POW]]
float4 test_pow_int64_t4(int64_t4 p0, int64_t4 p1) { return pow(p0, p1); }

// CHECK-LABEL: define hidden noundef nofpclass(nan inf) float {{.*}}test_pow_uint64_t
// CHECK: [[CONV0:%.*]] = uitofp i64 %{{.*}} to float
// CHECK: [[CONV1:%.*]] = uitofp i64 %{{.*}} to float
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef float @llvm.pow.f32(float [[CONV0]], float [[CONV1]])
// CHECK: ret float [[POW]]
float test_pow_uint64_t(uint64_t p0, uint64_t p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <2 x float> {{.*}}test_pow_uint64_t2
// CHECK: [[CONV0:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK: [[CONV1:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <2 x float> @llvm.pow.v2f32(<2 x float> [[CONV0]], <2 x float> [[CONV1]])
// CHECK: ret <2 x float> [[POW]]
float2 test_pow_uint64_t2(uint64_t2 p0, uint64_t2 p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <3 x float> {{.*}}test_pow_uint64_t3
// CHECK: [[CONV0:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK: [[CONV1:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <3 x float> @llvm.pow.v3f32(<3 x float> [[CONV0]], <3 x float> [[CONV1]])
// CHECK: ret <3 x float> [[POW]]
float3 test_pow_uint64_t3(uint64_t3 p0, uint64_t3 p1) { return pow(p0, p1); }
// CHECK-LABEL: define hidden noundef nofpclass(nan inf) <4 x float> {{.*}}test_pow_uint64_t4
// CHECK: [[CONV0:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK: [[CONV1:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK: [[POW:%.*]] = call [[FLOATATTRS]] noundef <4 x float> @llvm.pow.v4f32(<4 x float> [[CONV0]], <4 x float> [[CONV1]])
// CHECK: ret <4 x float> [[POW]]
float4 test_pow_uint64_t4(uint64_t4 p0, uint64_t4 p1) { return pow(p0, p1); }
