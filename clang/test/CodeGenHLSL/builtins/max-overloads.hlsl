// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF-LABEL: define noundef <4 x i16> {{.*}}test_max_short4_mismatch
// NATIVE_HALF: call <4 x i16> @llvm.smax.v4i16
int16_t4 test_max_short4_mismatch(int16_t4 p0, int16_t p1) { return max(p0, p1); }

// NATIVE_HALF-LABEL: define noundef <4 x i16> {{.*}}test_max_ushort4_mismatch
// NATIVE_HALF: call <4 x i16> @llvm.umax.v4i16
uint16_t4 test_max_ushort4_mismatch(uint16_t4 p0, uint16_t p1) { return max(p0, p1); }
#endif

// CHECK-LABEL: define noundef <4 x i32> {{.*}}test_max_int4_mismatch
// CHECK: call <4 x i32> @llvm.smax.v4i32
int4 test_max_int4_mismatch(int4 p0, int p1) { return max(p0, p1); }

// CHECK-LABEL: define noundef <4 x i32> {{.*}}test_max_uint4_mismatch
// CHECK: call <4 x i32> @llvm.umax.v4i32
uint4 test_max_uint4_mismatch(uint4 p0, uint p1) { return max(p0, p1); }

// CHECK-LABEL: define noundef <4 x i64> {{.*}}test_max_long4_mismatch
// CHECK: call <4 x i64> @llvm.smax.v4i64
int64_t4 test_max_long4_mismatch(int64_t4 p0, int64_t p1) { return max(p0, p1); }

// CHECK-LABEL: define noundef <4 x i64> {{.*}}test_max_ulong4_mismatch
// CHECK: call <4 x i64> @llvm.umax.v4i64
uint64_t4 test_max_ulong4_mismatch(uint64_t4 p0, uint64_t p1) { return max(p0, p1); }

// NATIVE_HALF-LABEL: define noundef nofpclass(nan inf) <4 x half> {{.*}}test_max_half4_mismatch
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.maxnum.v4f16
// NO_HALF-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_max_half4_mismatch
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.maxnum.v4f32(
half4 test_max_half4_mismatch(half4 p0, half p1) { return max(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x float> {{.*}}test_max_float4_mismatch
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.maxnum.v4f32
float4 test_max_float4_mismatch(float4 p0, float p1) { return max(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x double> {{.*}}test_max_double4_mismatch
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.maxnum.v4f64
double4 test_max_double4_mismatch(double4 p0, double p1) { return max(p0, p1); }

// CHECK-LABEL: define noundef nofpclass(nan inf) <4 x double> {{.*}}test_max_double4_mismatch2
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.maxnum.v4f64
double4 test_max_double4_mismatch2(double4 p0, double p1) { return max(p1, p0); }
