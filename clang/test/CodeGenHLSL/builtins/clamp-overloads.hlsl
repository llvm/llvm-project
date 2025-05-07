// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF \
// RUN:  -DTARGET=dx -DFNATTRS=noundef -DFFNATTRS="nofpclass(nan inf)"
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:  -DTARGET=dx -DFNATTRS=noundef -DFFNATTRS="nofpclass(nan inf)"
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple spirv-unknown-vulkan-compute %s \
// RUN:  -fnative-half-type -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NATIVE_HALF \
// RUN:  -DTARGET=spv -DFNATTRS="spir_func noundef" -DFFNATTRS="nofpclass(nan inf)"
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple spirv-unknown-vulkan-compute %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | \
// RUN:  FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:  -DTARGET=spv -DFNATTRS="spir_func noundef" -DFFNATTRS="nofpclass(nan inf)"

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF: define [[FNATTRS]] <4 x i16> {{.*}}test_clamp_short4_mismatch
// NATIVE_HALF: call <4 x i16> @llvm.[[TARGET]].sclamp.v4i16
int16_t4 test_clamp_short4_mismatch(int16_t4 p0, int16_t p1) { return clamp(p0, p0,p1); }

// NATIVE_HALF: define [[FNATTRS]] <4 x i16> {{.*}}test_clamp_ushort4_mismatch
// NATIVE_HALF: call <4 x i16> @llvm.[[TARGET]].uclamp.v4i16
uint16_t4 test_clamp_ushort4_mismatch(uint16_t4 p0, uint16_t p1) { return clamp(p0, p0,p1); }
#endif

// CHECK: define [[FNATTRS]] <4 x i32> {{.*}}test_clamp_int4_mismatch
// CHECK: call <4 x i32> @llvm.[[TARGET]].sclamp.v4i32
int4 test_clamp_int4_mismatch(int4 p0, int p1) { return clamp(p0, p0,p1); }

// CHECK: define [[FNATTRS]] <4 x i32> {{.*}}test_clamp_uint4_mismatch
// CHECK: call <4 x i32> @llvm.[[TARGET]].uclamp.v4i32
uint4 test_clamp_uint4_mismatch(uint4 p0, uint p1) { return clamp(p0, p0,p1); }

// CHECK: define [[FNATTRS]] <4 x i64> {{.*}}test_clamp_long4_mismatch
// CHECK: call <4 x i64> @llvm.[[TARGET]].sclamp.v4i64
int64_t4 test_clamp_long4_mismatch(int64_t4 p0, int64_t p1) { return clamp(p0, p0,p1); }

// CHECK: define [[FNATTRS]] <4 x i64> {{.*}}test_clamp_ulong4_mismatch
// CHECK: call <4 x i64> @llvm.[[TARGET]].uclamp.v4i64
uint64_t4 test_clamp_ulong4_mismatch(uint64_t4 p0, uint64_t p1) { return clamp(p0, p0,p1); }

// NATIVE_HALF: define [[FNATTRS]] [[FFNATTRS]] <4 x half> {{.*}}test_clamp_half4_mismatch
// NATIVE_HALF: call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.[[TARGET]].nclamp.v4f16
// NO_HALF: define [[FNATTRS]] [[FFNATTRS]] <4 x float> {{.*}}test_clamp_half4_mismatch
// NO_HALF: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].nclamp.v4f32(
half4 test_clamp_half4_mismatch(half4 p0, half p1) { return clamp(p0, p0,p1); }

// CHECK: define [[FNATTRS]] [[FFNATTRS]] <4 x float> {{.*}}test_clamp_float4_mismatch
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].nclamp.v4f32
float4 test_clamp_float4_mismatch(float4 p0, float p1) { return clamp(p0, p0,p1); }

// CHECK: define [[FNATTRS]] [[FFNATTRS]] <4 x double> {{.*}}test_clamp_double4_mismatch
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.[[TARGET]].nclamp.v4f64
double4 test_clamp_double4_mismatch(double4 p0, double p1) { return clamp(p0, p0,p1); }
// CHECK: define [[FNATTRS]] [[FFNATTRS]] <4 x double> {{.*}}test_clamp_double4_mismatch2
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.[[TARGET]].nclamp.v4f64
double4 test_clamp_double4_mismatch2(double4 p0, double p1) { return clamp(p0, p1,p0); }

// CHECK: define [[FNATTRS]] <3 x i32> {{.*}}test_overloads3
// CHECK: call <3 x i32> @llvm.[[TARGET]].uclamp.v3i32
uint3 test_overloads3(uint3 p0, uint p1, uint p2) { return clamp(p0, p1, p2); }
