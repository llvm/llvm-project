// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -fnative-half-type -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,NATIVE_HALF \
// RUN:  -DTARGET=dx -DFNATTRS=noundef -DFFNATTRS="nofpclass(nan inf)"

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:  -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:  -DTARGET=dx -DFNATTRS=noundef -DFFNATTRS="nofpclass(nan inf)"

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple spirv-unknown-vulkan-compute %s \
// RUN:  -fnative-half-type -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,NATIVE_HALF \
// RUN:  -DTARGET=spv -DFNATTRS="spir_func noundef" -DFFNATTRS="nofpclass(nan inf)"

// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -triple spirv-unknown-vulkan-compute %s \
// RUN:  -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:  -DTARGET=spv -DFNATTRS="spir_func noundef" -DFFNATTRS="nofpclass(nan inf)"

#ifdef __HLSL_ENABLE_16_BIT
// NATIVE_HALF: define [[FNATTRS]] <4 x i16> {{.*}}test_clamp_short4_mismatch
// NATIVE_HALF: [[CONV0:%.*]] = insertelement <4 x i16> poison, i16 %{{.*}}, i64 0
// NATIVE_HALF: [[CONV1:%.*]] = shufflevector <4 x i16> [[CONV0]], <4 x i16> poison, <4 x i32> zeroinitializer
// NATIVE_HALF: [[CLAMP:%.*]] = call {{.*}} <4 x i16> @llvm.[[TARGET]].sclamp.v4i16(<4 x i16> %{{.*}}, <4 x i16> %{{.*}}, <4 x i16> [[CONV1]])
// NATIVE_HALF: ret <4 x i16> [[CLAMP]]
int16_t4 test_clamp_short4_mismatch(int16_t4 p0, int16_t p1) { return clamp(p0, p0,p1); }

// NATIVE_HALF: define [[FNATTRS]] <4 x i16> {{.*}}test_clamp_ushort4_mismatch
// NATIVE_HALF: [[CONV0:%.*]] = insertelement <4 x i16> poison, i16 %{{.*}}, i64 0
// NATIVE_HALF: [[CONV1:%.*]] = shufflevector <4 x i16> [[CONV0]], <4 x i16> poison, <4 x i32> zeroinitializer
// NATIVE_HALF: [[CLAMP:%.*]] = call {{.*}} <4 x i16> @llvm.[[TARGET]].uclamp.v4i16(<4 x i16> %{{.*}}, <4 x i16> %{{.*}}, <4 x i16> [[CONV1]])
// NATIVE_HALF: ret <4 x i16> [[CLAMP]]
uint16_t4 test_clamp_ushort4_mismatch(uint16_t4 p0, uint16_t p1) { return clamp(p0, p0,p1); }
#endif

// CHECK: define [[FNATTRS]] <4 x i32> {{.*}}test_clamp_int4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i32> poison, i32 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i32> [[CONV0]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[CLAMP:%.*]] = call {{.*}} <4 x i32> @llvm.[[TARGET]].sclamp.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> [[CONV1]])
// CHECK: ret <4 x i32> [[CLAMP]]
int4 test_clamp_int4_mismatch(int4 p0, int p1) { return clamp(p0, p0,p1); }

// CHECK: define [[FNATTRS]] <4 x i32> {{.*}}test_clamp_uint4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i32> poison, i32 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i32> [[CONV0]], <4 x i32> poison, <4 x i32> zeroinitializer
// CHECK: [[CLAMP:%.*]] = call {{.*}} <4 x i32> @llvm.[[TARGET]].uclamp.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> [[CONV1]])
// CHECK: ret <4 x i32> [[CLAMP]]
uint4 test_clamp_uint4_mismatch(uint4 p0, uint p1) { return clamp(p0, p0,p1); }

// CHECK: define [[FNATTRS]] <4 x i64> {{.*}}test_clamp_long4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i64> poison, i64 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i64> [[CONV0]], <4 x i64> poison, <4 x i32> zeroinitializer
// CHECK: [[CLAMP:%.*]] = call {{.*}} <4 x i64> @llvm.[[TARGET]].sclamp.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> [[CONV1]])
// CHECK: ret <4 x i64> [[CLAMP]]
int64_t4 test_clamp_long4_mismatch(int64_t4 p0, int64_t p1) { return clamp(p0, p0,p1); }

// CHECK: define [[FNATTRS]] <4 x i64> {{.*}}test_clamp_ulong4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x i64> poison, i64 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x i64> [[CONV0]], <4 x i64> poison, <4 x i32> zeroinitializer
// CHECK: [[CLAMP:%.*]] = call {{.*}} <4 x i64> @llvm.[[TARGET]].uclamp.v4i64(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> [[CONV1]])
// CHECK: ret <4 x i64> [[CLAMP]]
uint64_t4 test_clamp_ulong4_mismatch(uint64_t4 p0, uint64_t p1) { return clamp(p0, p0,p1); }

// NATIVE_HALF: define [[FNATTRS]] [[FFNATTRS]] <4 x half> {{.*}}test_clamp_half4_mismatch
// NATIVE_HALF: [[CONV0:%.*]] = insertelement <4 x half> poison, half %{{.*}}, i64 0
// NATIVE_HALF: [[CONV1:%.*]] = shufflevector <4 x half> [[CONV0]], <4 x half> poison, <4 x i32> zeroinitializer
// NATIVE_HALF: [[CLAMP:%.*]] = call reassoc nnan ninf nsz arcp afn {{.*}} <4 x half> @llvm.[[TARGET]].nclamp.v4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <4 x half> [[CONV1]])
// NATIVE_HALF: ret <4 x half> [[CLAMP]]
// NO_HALF: define [[FNATTRS]] [[FFNATTRS]] <4 x float> {{.*}}test_clamp_half4_mismatch
// NO_HALF: [[CONV0:%.*]] = insertelement <4 x float> poison, float %{{.*}}, i64 0
// NO_HALF: [[CONV1:%.*]] = shufflevector <4 x float> [[CONV0]], <4 x float> poison, <4 x i32> zeroinitializer
// NO_HALF: [[CLAMP:%.*]] = call reassoc nnan ninf nsz arcp afn {{.*}} <4 x float> @llvm.[[TARGET]].nclamp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[CONV1]])
// NO_HALF: ret <4 x float> [[CLAMP]]
half4 test_clamp_half4_mismatch(half4 p0, half p1) { return clamp(p0, p0,p1); }

// CHECK: define [[FNATTRS]] [[FFNATTRS]] <4 x float> {{.*}}test_clamp_float4_mismatch
// CHECK: [[CONV0:%.*]] = insertelement <4 x float> poison, float %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x float> [[CONV0]], <4 x float> poison, <4 x i32> zeroinitializer
// CHECK: [[CLAMP:%.*]] = call reassoc nnan ninf nsz arcp afn {{.*}} <4 x float> @llvm.[[TARGET]].nclamp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> [[CONV1]])
// CHECK: ret <4 x float> [[CLAMP]]
float4 test_clamp_float4_mismatch(float4 p0, float p1) { return clamp(p0, p0,p1); }


// CHECK: define [[FNATTRS]] [[FFNATTRS]] <4 x double> {{.*}}test_clamp_double4_mismatch1
// CHECK: [[CONV0:%.*]] = insertelement <4 x double> poison, double %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x double> [[CONV0]], <4 x double> poison, <4 x i32> zeroinitializer
// CHECK: [[CLAMP:%.*]] = call reassoc nnan ninf nsz arcp afn {{.*}} <4 x double> @llvm.[[TARGET]].nclamp.v4f64(<4 x double> %{{.*}}, <4 x double> %{{.*}}, <4 x double> [[CONV1]])
// CHECK: ret <4 x double> [[CLAMP]]
double4 test_clamp_double4_mismatch1(double4 p0, double p1) { return clamp(p0, p0,p1); }
// CHECK: define [[FNATTRS]] [[FFNATTRS]] <4 x double> {{.*}}test_clamp_double4_mismatch2
// CHECK: [[CONV0:%.*]] = insertelement <4 x double> poison, double %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <4 x double> [[CONV0]], <4 x double> poison, <4 x i32> zeroinitializer
// CHECK: [[CLAMP:%.*]] = call reassoc nnan ninf nsz arcp afn {{.*}} <4 x double> @llvm.[[TARGET]].nclamp.v4f64(<4 x double> %{{.*}}, <4 x double> [[CONV1]], <4 x double> %{{.*}})
// CHECK: ret <4 x double> [[CLAMP]]
double4 test_clamp_double4_mismatch2(double4 p0, double p1) { return clamp(p0, p1,p0); }

// CHECK: define [[FNATTRS]] <3 x i32> {{.*}}test_overloads3
// CHECK: [[CONV0:%.*]] = insertelement <3 x i32> poison, i32 %{{.*}}, i64 0
// CHECK: [[CONV1:%.*]] = shufflevector <3 x i32> [[CONV0]], <3 x i32> poison, <3 x i32> zeroinitializer
// CHECK: [[CONV2:%.*]] = insertelement <3 x i32> poison, i32 %{{.*}}, i64 0
// CHECK: [[CONV3:%.*]] = shufflevector <3 x i32> [[CONV2]], <3 x i32> poison, <3 x i32> zeroinitializer
// CHECK: [[CLAMP:%.*]] = call {{.*}} <3 x i32> @llvm.[[TARGET]].uclamp.v3i32(<3 x i32> %{{.*}}, <3 x i32> [[CONV1]], <3 x i32> [[CONV3]])
// CHECK: ret <3 x i32> [[CLAMP]]
uint3 test_overloads3(uint3 p0, uint p1, uint p2) { return clamp(p0, p1, p2); }
