// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DFNATTRS="noundef nofpclass(nan inf)" -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DFNATTRS="noundef nofpclass(nan inf)" -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   --check-prefixes=CHECK,NATIVE_HALF \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTARGET=spv
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK,NO_HALF \
// RUN:   -DFNATTRS="spir_func noundef nofpclass(nan inf)" -DTARGET=spv

// NATIVE_HALF: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn half @llvm.[[TARGET]].lerp.f16(half %{{.*}}, half %{{.*}}, half %{{.*}})
// NATIVE_HALF: ret half %hlsl.lerp
// NO_HALF: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].lerp.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
// NO_HALF: ret float %hlsl.lerp
half test_lerp_half(half p0) { return lerp(p0, p0, p0); }

// NATIVE_HALF: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.[[TARGET]].lerp.v2f16(<2 x half> %{{.*}}, <2 x half> %{{.*}}, <2 x half> %{{.*}})
// NATIVE_HALF: ret <2 x half> %hlsl.lerp
// NO_HALF: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// NO_HALF: ret <2 x float> %hlsl.lerp
half2 test_lerp_half2(half2 p0) { return lerp(p0, p0, p0); }

// NATIVE_HALF: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.[[TARGET]].lerp.v3f16(<3 x half> %{{.*}}, <3 x half> %{{.*}}, <3 x half> %{{.*}})
// NATIVE_HALF: ret <3 x half> %hlsl.lerp
// NO_HALF: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}}, <3 x float> %{{.*}})
// NO_HALF: ret <3 x float> %hlsl.lerp
half3 test_lerp_half3(half3 p0) { return lerp(p0, p0, p0); }

// NATIVE_HALF: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.[[TARGET]].lerp.v4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}}, <4 x half> %{{.*}})
// NATIVE_HALF: ret <4 x half> %hlsl.lerp
// NO_HALF: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
// NO_HALF: ret <4 x float> %hlsl.lerp
half4 test_lerp_half4(half4 p0) { return lerp(p0, p0, p0); }

// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].lerp.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
// CHECK: ret float %hlsl.lerp
float test_lerp_float(float p0) { return lerp(p0, p0, p0); }

// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.lerp
float2 test_lerp_float2(float2 p0) { return lerp(p0, p0, p0); }

// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}}, <3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.lerp
float3 test_lerp_float3(float3 p0) { return lerp(p0, p0, p0); }

// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.lerp
float4 test_lerp_float4(float4 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_double
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].lerp.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
// CHECK: ret float %hlsl.lerp
float test_lerp_double(double p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_double2
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.lerp
float2 test_lerp_double2(double2 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_double3
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}}, <3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.lerp
float3 test_lerp_double3(double3 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_double4
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.lerp
float4 test_lerp_double4(double4 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_int
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].lerp.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
// CHECK: ret float %hlsl.lerp
float test_lerp_int(int p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_int2
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.lerp
float2 test_lerp_int2(int2 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_int3
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}}, <3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.lerp
float3 test_lerp_int3(int3 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_int4
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.lerp
float4 test_lerp_int4(int4 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_uint
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].lerp.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
// CHECK: ret float %hlsl.lerp
float test_lerp_uint(uint p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_uint2
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.lerp
float2 test_lerp_uint2(uint2 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_uint3
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}}, <3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.lerp
float3 test_lerp_uint3(uint3 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_uint4
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.lerp
float4 test_lerp_uint4(uint4 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_int64_t
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].lerp.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
// CHECK: ret float %hlsl.lerp
float test_lerp_int64_t(int64_t p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_int64_t2
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.lerp
float2 test_lerp_int64_t2(int64_t2 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_int64_t3
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}}, <3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.lerp
float3 test_lerp_int64_t3(int64_t3 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_int64_t4
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.lerp
float4 test_lerp_int64_t4(int64_t4 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_uint64_t
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn float @llvm.[[TARGET]].lerp.f32(float %{{.*}}, float %{{.*}}, float %{{.*}})
// CHECK: ret float %hlsl.lerp
float test_lerp_uint64_t(uint64_t p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_uint64_t2
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.[[TARGET]].lerp.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %{{.*}})
// CHECK: ret <2 x float> %hlsl.lerp
float2 test_lerp_uint64_t2(uint64_t2 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_uint64_t3
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.[[TARGET]].lerp.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}}, <3 x float> %{{.*}})
// CHECK: ret <3 x float> %hlsl.lerp
float3 test_lerp_uint64_t3(uint64_t3 p0) { return lerp(p0, p0, p0); }

// CHECK-LABEL: test_lerp_uint64_t4
// CHECK: %hlsl.lerp = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.[[TARGET]].lerp.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x float> %{{.*}})
// CHECK: ret <4 x float> %hlsl.lerp
float4 test_lerp_uint64_t4(uint64_t4 p0) { return lerp(p0, p0, p0); }
