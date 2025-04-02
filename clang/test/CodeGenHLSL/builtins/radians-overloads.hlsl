// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DTARGET=dx -DFNATTRS="noundef nofpclass(nan inf)"
// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK \
// RUN:   -DTARGET=spv -DFNATTRS="spir_func noundef nofpclass(nan inf)"

// CHECK: define [[FNATTRS]] float @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].radians.f32(
// CHECK: ret float %{{.*}}
float test_radians_double(double p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].radians.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_radians_double2(double2 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].radians.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_radians_double3(double3 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].radians.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_radians_double4(double4 p0) { return radians(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].radians.f32(
// CHECK: ret float %{{.*}}
float test_radians_int(int p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].radians.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_radians_int2(int2 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].radians.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_radians_int3(int3 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].radians.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_radians_int4(int4 p0) { return radians(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].radians.f32(
// CHECK: ret float %{{.*}}
float test_radians_uint(uint p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].radians.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_radians_uint2(uint2 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].radians.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_radians_uint3(uint3 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].radians.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_radians_uint4(uint4 p0) { return radians(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].radians.f32(
// CHECK: ret float %{{.*}}
float test_radians_int64_t(int64_t p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].radians.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_radians_int64_t2(int64_t2 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].radians.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_radians_int64_t3(int64_t3 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].radians.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_radians_int64_t4(int64_t4 p0) { return radians(p0); }

// CHECK: define [[FNATTRS]] float @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].radians.f32(
// CHECK: ret float %{{.*}}
float test_radians_uint64_t(uint64_t p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <2 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].radians.v2f32
// CHECK: ret <2 x float> %{{.*}}
float2 test_radians_uint64_t2(uint64_t2 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <3 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].radians.v3f32
// CHECK: ret <3 x float> %{{.*}}
float3 test_radians_uint64_t3(uint64_t3 p0) { return radians(p0); }
// CHECK: define [[FNATTRS]] <4 x float> @
// CHECK: %{{.*}} = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].radians.v4f32
// CHECK: ret <4 x float> %{{.*}}
float4 test_radians_uint64_t4(uint64_t4 p0) { return radians(p0); }
