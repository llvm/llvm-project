// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF-LABEL: define noundef <4 x i16> {{.*}}test_min_short4_mismatch
// NATIVE_HALF: [[CONV0:%.*]] = insertelement <4 x i16> poison, i16 %{{.*}}, i64 0
// NATIVE_HALF: [[CONV1:%.*]] = shufflevector <4 x i16> [[CONV0]], <4 x i16> poison, <4 x i32> zeroinitializer
// NATIVE_HALF: [[MIN:%.*]] = call noundef <4 x i16> @llvm.smin.v4i16(<4 x i16> %{{.*}}, <4 x i16> [[CONV1]])
// NATIVE_HALF: ret <4 x i16> [[MIN]]
int16_t4 test_min_short4_mismatch(int16_t4 p0, int16_t p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: define noundef <4 x i16> {{.*}}test_min_ushort4_mismatch
// NATIVE_HALF: [[CONV0:%.*]] = insertelement <4 x i16> poison, i16 %{{.*}}, i64 0
// NATIVE_HALF: [[CONV1:%.*]] = shufflevector <4 x i16> [[CONV0]], <4 x i16> poison, <4 x i32> zeroinitializer
// NATIVE_HALF: [[MIN:%.*]] = call noundef <4 x i16> @llvm.umin.v4i16(<4 x i16> %{{.*}}, <4 x i16> [[CONV1]])
// NATIVE_HALF: ret <4 x i16> [[MIN]]
uint16_t4 test_min_ushort4_mismatch(uint16_t4 p0, uint16_t p1) { return min(p0, p1); }
#endif

// CHECK-LABEL: define noundef <4 x i32> {{.*}}test_min_int4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i32> poison, i32 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i32> [[CONV0]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[MIN:%.*]] = call noundef <4 x i32> @llvm.smin.v4i32(<4 x i32> %{{.*}}, <4 x i32> [[CONV1]])
// CHECK: ret <4 x i32> [[MIN]]
int4 test_min_int4_mismatch(int4 p0, int p1) { return min(p0, p1); }

// CHECK-LABEL: define noundef <4 x i32> {{.*}}test_min_uint4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i32> poison, i32 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i32> [[CONV0]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[MIN:%.*]] = call noundef <4 x i32> @llvm.umin.v4i32(<4 x i32> %{{.*}}, <4 x i32> [[CONV1]])
// CHECK: ret <4 x i32> [[MIN]]
uint4 test_min_uint4_mismatch(uint4 p0, uint p1) { return min(p0, p1); }

// CHECK-LABEL: define noundef <4 x i64> {{.*}}test_min_long4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i64> poison, i64 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i64> [[CONV0]], <4 x i64> poison, <4 x i32> zeroinitializer
// CHECK: [[MIN:%.*]] = call noundef <4 x i64> @llvm.smin.v4i64(<4 x i64> %{{.*}}, <4 x i64> [[CONV1]])
// CHECK: ret <4 x i64> [[MIN]]
int64_t4 test_min_long4_mismatch(int64_t4 p0, int64_t p1) { return min(p0, p1); }

// CHECK-LABEL: define noundef <4 x i64> {{.*}}test_min_ulong4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i64> poison, i64 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i64> [[CONV0]], <4 x i64> poison, <4 x i32> zeroinitializer
// CHECK: [[MIN:%.*]] = call noundef <4 x i64> @llvm.umin.v4i64(<4 x i64> %{{.*}}, <4 x i64> [[CONV1]])
// CHECK: ret <4 x i64> [[MIN]]
uint64_t4 test_min_ulong4_mismatch(uint64_t4 p0, uint64_t p1) { return min(p0, p1); }

// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <4 x half> {{.*}}test_min_half4_mismatch
// NATIVE_HALF: [[CONV0:%.*]] = insertelement <4 x half> poison, half %{{.*}}, i64 0
// NATIVE_HALF: [[CONV1:%.*]] = shufflevector <4 x half> [[CONV0]], <4 x half> poison, <4 x i32> zeroinitializer
// NATIVE_HALF: [[MIN:%.*]] = call reassoc nnan ninf nsz arcp afn noundef <4 x half> @llvm.minnum.v4f16(<4 x half> %{{.*}}, <4 x half> [[CONV1]])
// NATIVE_HALF: ret <4 x half> [[MIN]]
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_min_half4_mismatch
// NO_HALF: [[CONV0:%.*]] = insertelement <4 x float> poison, float %{{.*}}, i64 0
// NO_HALF: [[CONV1:%.*]] = shufflevector <4 x float> [[CONV0]], <4 x float> poison, <4 x i32> zeroinitializer
// NO_HALF: [[MIN:%.*]] = call reassoc nnan ninf nsz arcp afn noundef <4 x float> @llvm.minnum.v4f32(<4 x float> %{{.*}}, <4 x float> [[CONV1]])
// NO_HALF: ret <4 x float> [[MIN]]
half4 test_min_half4_mismatch(half4 p0, half p1) { return min(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_min_float4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x float> poison, float %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x float> [[CONV0]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: [[MIN:%.*]] = call reassoc nnan ninf nsz arcp afn noundef <4 x float> @llvm.minnum.v4f32(<4 x float> %{{.*}}, <4 x float> [[CONV1]])
// CHECK: ret <4 x float> [[MIN]]
float4 test_min_float4_mismatch(float4 p0, float p1) { return min(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x double> {{.*}}test_min_double4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x double> poison, double %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x double> [[CONV0]], <4 x double> poison, <4 x i32> zeroinitializer
// CHECK: [[MIN:%.*]] = call reassoc nnan ninf nsz arcp afn noundef <4 x double> @llvm.minnum.v4f64(<4 x double> %{{.*}}, <4 x double> [[CONV1]])
// CHECK: ret <4 x double> [[MIN]]
double4 test_min_double4_mismatch(double4 p0, double p1) { return min(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x double> {{.*}}test_min_double4_mismatch2
// CHECK: [[CONV0:%.*]] = insertelement <4 x double> poison, double %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x double> [[CONV0]], <4 x double> poison, <4 x i32> zeroinitializer
// CHECK: [[MIN:%.*]] = call reassoc nnan ninf nsz arcp afn noundef <4 x double> @llvm.minnum.v4f64(<4 x double> [[CONV1]], <4 x double> %{{.*}})
// CHECK: ret <4 x double> [[MIN]]
double4 test_min_double4_mismatch2(double4 p0, double p1) { return min(p1, p0); }
