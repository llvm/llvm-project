// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -fnative-int16-type -emit-llvm  -o - | FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF-LABEL: define hidden noundef <4 x i16> {{.*}}test_max_short4_mismatch
// NATIVE_HALF: [[CONV0:%.*]] = insertelement <4 x i16> poison, i16 %{{.*}}, i64 0
// NATIVE_HALF: [[CONV1:%.*]] = shufflevector <4 x i16> [[CONV0]], <4 x i16> poison, <4 x i32> zeroinitializer
// NATIVE_HALF: [[MAX:%.*]] = call noundef <4 x i16> @llvm.smax.v4i16(<4 x i16> %{{.*}}, <4 x i16> [[CONV1]])
// NATIVE_HALF: ret <4 x i16> [[MAX]]
int16_t4 test_max_short4_mismatch(int16_t4 p0, int16_t p1) { return max(p0, p1); }

// NATIVE_HALF-LABEL: define hidden noundef <4 x i16> {{.*}}test_max_ushort4_mismatch
// NATIVE_HALF: [[CONV0:%.*]] = insertelement <4 x i16> poison, i16 %{{.*}}, i64 0
// NATIVE_HALF: [[CONV1:%.*]] = shufflevector <4 x i16> [[CONV0]], <4 x i16> poison, <4 x i32> zeroinitializer
// NATIVE_HALF: [[MAX:%.*]] = call noundef <4 x i16> @llvm.umax.v4i16(<4 x i16> %{{.*}}, <4 x i16> [[CONV1]])
// NATIVE_HALF: ret <4 x i16> [[MAX]]
uint16_t4 test_max_ushort4_mismatch(uint16_t4 p0, uint16_t p1) { return max(p0, p1); }
#endif

// CHECK-LABEL: define hidden noundef <4 x i32> {{.*}}test_max_int4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i32> poison, i32 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i32> [[CONV0]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[MAX:%.*]] = call noundef <4 x i32> @llvm.smax.v4i32(<4 x i32> %{{.*}}, <4 x i32> [[CONV1]])
// CHECK: ret <4 x i32> [[MAX]]
int4 test_max_int4_mismatch(int4 p0, int p1) { return max(p0, p1); }

// CHECK-LABEL: define hidden noundef <4 x i32> {{.*}}test_max_uint4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i32> poison, i32 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i32> [[CONV0]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[MAX:%.*]] = call noundef <4 x i32> @llvm.umax.v4i32(<4 x i32> %{{.*}}, <4 x i32> [[CONV1]])
// CHECK: ret <4 x i32> [[MAX]]
uint4 test_max_uint4_mismatch(uint4 p0, uint p1) { return max(p0, p1); }

// CHECK-LABEL: define hidden noundef <4 x i64> {{.*}}test_max_long4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i64> poison, i64 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i64> [[CONV0]], <4 x i64> poison, <4 x i32> zeroinitializer
// CHECK: [[MAX:%.*]] = call noundef <4 x i64> @llvm.smax.v4i64(<4 x i64> %{{.*}}, <4 x i64> [[CONV1]])
// CHECK: ret <4 x i64> [[MAX]]
int64_t4 test_max_long4_mismatch(int64_t4 p0, int64_t p1) { return max(p0, p1); }

// CHECK-LABEL: define hidden noundef <4 x i64> {{.*}}test_max_ulong4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i64> poison, i64 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i64> [[CONV0]], <4 x i64> poison, <4 x i32> zeroinitializer
// CHECK: [[MAX:%.*]] = call noundef <4 x i64> @llvm.umax.v4i64(<4 x i64> %{{.*}}, <4 x i64> [[CONV1]])
// CHECK: ret <4 x i64> [[MAX]]
uint64_t4 test_max_ulong4_mismatch(uint64_t4 p0, uint64_t p1) { return max(p0, p1); }
