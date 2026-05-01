// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm  \
// RUN:   -o - | FileCheck %s --check-prefixes=CHECK -DFNATTRS="hidden spir_func noundef nofpclass(nan inf)" 

// CHECK: define [[FNATTRS]] <2 x float> @_{{.*}}test_atan2_double1x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <2 x float> @llvm.atan2.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[V5]]
float1x2 test_atan2_double1x2 (double1x2 p0, double1x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <3 x float> @_{{.*}}test_atan2_double1x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <3 x float> @llvm.atan2.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[V5]]
float1x3 test_atan2_double1x3 (double1x3 p0, double1x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_double1x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float1x4 test_atan2_double1x4 (double1x4 p0, double1x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <2 x float> @_{{.*}}test_atan2_double2x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <2 x double> %{{.*}} to <2 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <2 x float> @llvm.atan2.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[V5]]
float2x1 test_atan2_double2x1 (double2x1 p0, double2x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_double2x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float2x2 test_atan2_double2x2 (double2x2 p0, double2x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <6 x float> @_{{.*}}test_atan2_double2x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <6 x double> %{{.*}} to <6 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <6 x double> %{{.*}} to <6 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <6 x float> @llvm.atan2.v6f32(<6 x float> [[CONVI]], <6 x float> [[CONV1I]])
// CHECK:    ret <6 x float> [[V5]]
float2x3 test_atan2_double2x3 (double2x3 p0, double2x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <8 x float> @_{{.*}}test_atan2_double2x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <8 x double> %{{.*}} to <8 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <8 x double> %{{.*}} to <8 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <8 x float> @llvm.atan2.v8f32(<8 x float> [[CONVI]], <8 x float> [[CONV1I]])
// CHECK:    ret <8 x float> [[V5]]
float2x4 test_atan2_double2x4 (double2x4 p0, double2x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <3 x float> @_{{.*}}test_atan2_double3x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <3 x double> %{{.*}} to <3 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <3 x float> @llvm.atan2.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[V5]]
float3x1 test_atan2_double3x1 (double3x1 p0, double3x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <6 x float> @_{{.*}}test_atan2_double3x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <6 x double> %{{.*}} to <6 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <6 x double> %{{.*}} to <6 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <6 x float> @llvm.atan2.v6f32(<6 x float> [[CONVI]], <6 x float> [[CONV1I]])
// CHECK:    ret <6 x float> [[V5]]
float3x2 test_atan2_double3x2 (double3x2 p0, double3x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <9 x float> @_{{.*}}test_atan2_double3x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <9 x double> %{{.*}} to <9 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <9 x double> %{{.*}} to <9 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <9 x float> @llvm.atan2.v9f32(<9 x float> [[CONVI]], <9 x float> [[CONV1I]])
// CHECK:    ret <9 x float> [[V5]]
float3x3 test_atan2_double3x3 (double3x3 p0, double3x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <12 x float> @_{{.*}}test_atan2_double3x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <12 x double> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <12 x double> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float3x4 test_atan2_double3x4 (double3x4 p0, double3x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_double4x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <4 x double> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float4x1 test_atan2_double4x1 (double4x1 p0, double4x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <8 x float> @_{{.*}}test_atan2_double4x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <8 x double> %{{.*}} to <8 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <8 x double> %{{.*}} to <8 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <8 x float> @llvm.atan2.v8f32(<8 x float> [[CONVI]], <8 x float> [[CONV1I]])
// CHECK:    ret <8 x float> [[V5]]
float4x2 test_atan2_double4x2 (double4x2 p0, double4x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <12 x float> @_{{.*}}test_atan2_double4x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <12 x double> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <12 x double> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float4x3 test_atan2_double4x3 (double4x3 p0, double4x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <16 x float> @_{{.*}}test_atan2_double4x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = fptrunc {{.*}} <16 x double> %{{.*}} to <16 x float>
// CHECK:    [[CONV1I:%.*]] = fptrunc {{.*}} <16 x double> %{{.*}} to <16 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <16 x float> @llvm.atan2.v16f32(<16 x float> [[CONVI]], <16 x float> [[CONV1I]])
// CHECK:    ret <16 x float> [[V5]]
float4x4 test_atan2_double4x4 (double4x4 p0, double4x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <2 x float> @_{{.*}}test_atan2_uint1x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <2 x float> @llvm.atan2.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[V5]]
float1x2 test_atan2_uint1x2 (uint1x2 p0, uint1x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <3 x float> @_{{.*}}test_atan2_uint1x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <3 x float> @llvm.atan2.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[V5]]
float1x3 test_atan2_uint1x3 (uint1x3 p0, uint1x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_uint1x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float1x4 test_atan2_uint1x4 (uint1x4 p0, uint1x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <2 x float> @_{{.*}}test_atan2_uint2x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <2 x float> @llvm.atan2.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[V5]]
float2x1 test_atan2_uint2x1 (uint2x1 p0, uint2x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_uint2x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float2x2 test_atan2_uint2x2 (uint2x2 p0, uint2x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <6 x float> @_{{.*}}test_atan2_uint2x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <6 x i32> %{{.*}} to <6 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <6 x i32> %{{.*}} to <6 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <6 x float> @llvm.atan2.v6f32(<6 x float> [[CONVI]], <6 x float> [[CONV1I]])
// CHECK:    ret <6 x float> [[V5]]
float2x3 test_atan2_uint2x3 (uint2x3 p0, uint2x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <8 x float> @_{{.*}}test_atan2_uint2x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <8 x i32> %{{.*}} to <8 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <8 x i32> %{{.*}} to <8 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <8 x float> @llvm.atan2.v8f32(<8 x float> [[CONVI]], <8 x float> [[CONV1I]])
// CHECK:    ret <8 x float> [[V5]]
float2x4 test_atan2_uint2x4 (uint2x4 p0, uint2x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <3 x float> @_{{.*}}test_atan2_uint3x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <3 x float> @llvm.atan2.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[V5]]
float3x1 test_atan2_uint3x1 (uint3x1 p0, uint3x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <6 x float> @_{{.*}}test_atan2_uint3x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <6 x i32> %{{.*}} to <6 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <6 x i32> %{{.*}} to <6 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <6 x float> @llvm.atan2.v6f32(<6 x float> [[CONVI]], <6 x float> [[CONV1I]])
// CHECK:    ret <6 x float> [[V5]]
float3x2 test_atan2_uint3x2 (uint3x2 p0, uint3x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <9 x float> @_{{.*}}test_atan2_uint3x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <9 x i32> %{{.*}} to <9 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <9 x i32> %{{.*}} to <9 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <9 x float> @llvm.atan2.v9f32(<9 x float> [[CONVI]], <9 x float> [[CONV1I]])
// CHECK:    ret <9 x float> [[V5]]
float3x3 test_atan2_uint3x3 (uint3x3 p0, uint3x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <12 x float> @_{{.*}}test_atan2_uint3x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float3x4 test_atan2_uint3x4 (uint3x4 p0, uint3x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_uint4x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float4x1 test_atan2_uint4x1 (uint4x1 p0, uint4x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <8 x float> @_{{.*}}test_atan2_uint4x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <8 x i32> %{{.*}} to <8 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <8 x i32> %{{.*}} to <8 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <8 x float> @llvm.atan2.v8f32(<8 x float> [[CONVI]], <8 x float> [[CONV1I]])
// CHECK:    ret <8 x float> [[V5]]
float4x2 test_atan2_uint4x2 (uint4x2 p0, uint4x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <12 x float> @_{{.*}}test_atan2_uint4x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float4x3 test_atan2_uint4x3 (uint4x3 p0, uint4x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <16 x float> @_{{.*}}test_atan2_uint4x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <16 x i32> %{{.*}} to <16 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <16 x i32> %{{.*}} to <16 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <16 x float> @llvm.atan2.v16f32(<16 x float> [[CONVI]], <16 x float> [[CONV1I]])
// CHECK:    ret <16 x float> [[V5]]
float4x4 test_atan2_uint4x4 (uint4x4 p0, uint4x4 p1) {
  return atan2(p0, p1);
}


// CHECK: define [[FNATTRS]] <2 x float> @_{{.*}}test_atan2_int1x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <2 x float> @llvm.atan2.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[V5]]
float1x2 test_atan2_int1x2 (int1x2 p0, int1x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <3 x float> @_{{.*}}test_atan2_int1x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <3 x float> @llvm.atan2.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[V5]]
float1x3 test_atan2_int1x3 (int1x3 p0, int1x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_int1x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float1x4 test_atan2_int1x4 (int1x4 p0, int1x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <2 x float> @_{{.*}}test_atan2_int2x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <2 x i32> %{{.*}} to <2 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <2 x float> @llvm.atan2.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[V5]]
float2x1 test_atan2_int2x1 (int2x1 p0, int2x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_int2x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float2x2 test_atan2_int2x2 (int2x2 p0, int2x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <6 x float> @_{{.*}}test_atan2_int2x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <6 x i32> %{{.*}} to <6 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <6 x i32> %{{.*}} to <6 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <6 x float> @llvm.atan2.v6f32(<6 x float> [[CONVI]], <6 x float> [[CONV1I]])
// CHECK:    ret <6 x float> [[V5]]
float2x3 test_atan2_int2x3 (int2x3 p0, int2x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <8 x float> @_{{.*}}test_atan2_int2x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <8 x i32> %{{.*}} to <8 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <8 x i32> %{{.*}} to <8 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <8 x float> @llvm.atan2.v8f32(<8 x float> [[CONVI]], <8 x float> [[CONV1I]])
// CHECK:    ret <8 x float> [[V5]]
float2x4 test_atan2_int2x4 (int2x4 p0, int2x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <3 x float> @_{{.*}}test_atan2_int3x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <3 x i32> %{{.*}} to <3 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <3 x float> @llvm.atan2.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[V5]]
float3x1 test_atan2_int3x1 (int3x1 p0, int3x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <6 x float> @_{{.*}}test_atan2_int3x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <6 x i32> %{{.*}} to <6 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <6 x i32> %{{.*}} to <6 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <6 x float> @llvm.atan2.v6f32(<6 x float> [[CONVI]], <6 x float> [[CONV1I]])
// CHECK:    ret <6 x float> [[V5]]
float3x2 test_atan2_int3x2 (int3x2 p0, int3x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <9 x float> @_{{.*}}test_atan2_int3x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <9 x i32> %{{.*}} to <9 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <9 x i32> %{{.*}} to <9 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <9 x float> @llvm.atan2.v9f32(<9 x float> [[CONVI]], <9 x float> [[CONV1I]])
// CHECK:    ret <9 x float> [[V5]]
float3x3 test_atan2_int3x3 (int3x3 p0, int3x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <12 x float> @_{{.*}}test_atan2_int3x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float3x4 test_atan2_int3x4 (int3x4 p0, int3x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_int4x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <4 x i32> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float4x1 test_atan2_int4x1 (int4x1 p0, int4x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <8 x float> @_{{.*}}test_atan2_int4x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <8 x i32> %{{.*}} to <8 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <8 x i32> %{{.*}} to <8 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <8 x float> @llvm.atan2.v8f32(<8 x float> [[CONVI]], <8 x float> [[CONV1I]])
// CHECK:    ret <8 x float> [[V5]]
float4x2 test_atan2_int4x2 (int4x2 p0, int4x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <12 x float> @_{{.*}}test_atan2_int4x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <12 x i32> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float4x3 test_atan2_int4x3 (int4x3 p0, int4x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <16 x float> @_{{.*}}test_atan2_int4x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <16 x i32> %{{.*}} to <16 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <16 x i32> %{{.*}} to <16 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <16 x float> @llvm.atan2.v16f32(<16 x float> [[CONVI]], <16 x float> [[CONV1I]])
// CHECK:    ret <16 x float> [[V5]]
float4x4 test_atan2_int4x4 (int4x4 p0, int4x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <2 x float> @_{{.*}}test_atan2_int64_t1x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <2 x float> @llvm.atan2.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[V5]]
float1x2 test_atan2_int64_t1x2 (int64_t1x2 p0, int64_t1x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <3 x float> @_{{.*}}test_atan2_int64_t1x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <3 x float> @llvm.atan2.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[V5]]
float1x3 test_atan2_int64_t1x3 (int64_t1x3 p0, int64_t1x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_int64_t1x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float1x4 test_atan2_int64_t1x4 (int64_t1x4 p0, int64_t1x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <2 x float> @_{{.*}}test_atan2_int64_t2x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <2 x float> @llvm.atan2.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[V5]]
float2x1 test_atan2_int64_t2x1 (int64_t2x1 p0, int64_t2x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_int64_t2x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float2x2 test_atan2_int64_t2x2 (int64_t2x2 p0, int64_t2x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <6 x float> @_{{.*}}test_atan2_int64_t2x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <6 x i64> %{{.*}} to <6 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <6 x i64> %{{.*}} to <6 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <6 x float> @llvm.atan2.v6f32(<6 x float> [[CONVI]], <6 x float> [[CONV1I]])
// CHECK:    ret <6 x float> [[V5]]
float2x3 test_atan2_int64_t2x3 (int64_t2x3 p0, int64_t2x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <8 x float> @_{{.*}}test_atan2_int64_t2x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <8 x i64> %{{.*}} to <8 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <8 x i64> %{{.*}} to <8 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <8 x float> @llvm.atan2.v8f32(<8 x float> [[CONVI]], <8 x float> [[CONV1I]])
// CHECK:    ret <8 x float> [[V5]]
float2x4 test_atan2_int64_t2x4 (int64_t2x4 p0, int64_t2x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <3 x float> @_{{.*}}test_atan2_int64_t3x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <3 x float> @llvm.atan2.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[V5]]
float3x1 test_atan2_int64_t3x1 (int64_t3x1 p0, int64_t3x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <6 x float> @_{{.*}}test_atan2_int64_t3x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <6 x i64> %{{.*}} to <6 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <6 x i64> %{{.*}} to <6 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <6 x float> @llvm.atan2.v6f32(<6 x float> [[CONVI]], <6 x float> [[CONV1I]])
// CHECK:    ret <6 x float> [[V5]]
float3x2 test_atan2_int64_t3x2 (int64_t3x2 p0, int64_t3x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <9 x float> @_{{.*}}test_atan2_int64_t3x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <9 x i64> %{{.*}} to <9 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <9 x i64> %{{.*}} to <9 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <9 x float> @llvm.atan2.v9f32(<9 x float> [[CONVI]], <9 x float> [[CONV1I]])
// CHECK:    ret <9 x float> [[V5]]
float3x3 test_atan2_int64_t3x3 (int64_t3x3 p0, int64_t3x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <12 x float> @_{{.*}}test_atan2_int64_t3x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float3x4 test_atan2_int64_t3x4 (int64_t3x4 p0, int64_t3x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_int64_t4x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float4x1 test_atan2_int64_t4x1 (int64_t4x1 p0, int64_t4x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <8 x float> @_{{.*}}test_atan2_int64_t4x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <8 x i64> %{{.*}} to <8 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <8 x i64> %{{.*}} to <8 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <8 x float> @llvm.atan2.v8f32(<8 x float> [[CONVI]], <8 x float> [[CONV1I]])
// CHECK:    ret <8 x float> [[V5]]
float4x2 test_atan2_int64_t4x2 (int64_t4x2 p0, int64_t4x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <12 x float> @_{{.*}}test_atan2_int64_t4x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float4x3 test_atan2_int64_t4x3 (int64_t4x3 p0, int64_t4x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <16 x float> @_{{.*}}test_atan2_int64_t4x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = sitofp <16 x i64> %{{.*}} to <16 x float>
// CHECK:    [[CONV1I:%.*]] = sitofp <16 x i64> %{{.*}} to <16 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <16 x float> @llvm.atan2.v16f32(<16 x float> [[CONVI]], <16 x float> [[CONV1I]])
// CHECK:    ret <16 x float> [[V5]]
float4x4 test_atan2_int64_t4x4 (int64_t4x4 p0, int64_t4x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <2 x float> @_{{.*}}test_atan2_uint64_t1x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <2 x float> @llvm.atan2.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[V5]]
float1x2 test_atan2_uint64_t1x2 (uint64_t1x2 p0, uint64_t1x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <3 x float> @_{{.*}}test_atan2_uint64_t1x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <3 x float> @llvm.atan2.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[V5]]
float1x3 test_atan2_uint64_t1x3 (uint64_t1x3 p0, uint64_t1x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_uint64_t1x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float1x4 test_atan2_uint64_t1x4 (uint64_t1x4 p0, uint64_t1x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <2 x float> @_{{.*}}test_atan2_uint64_t2x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <2 x i64> %{{.*}} to <2 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <2 x float> @llvm.atan2.v2f32(<2 x float> [[CONVI]], <2 x float> [[CONV1I]])
// CHECK:    ret <2 x float> [[V5]]
float2x1 test_atan2_uint64_t2x1 (uint64_t2x1 p0, uint64_t2x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_uint64_t2x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float2x2 test_atan2_uint64_t2x2 (uint64_t2x2 p0, uint64_t2x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <6 x float> @_{{.*}}test_atan2_uint64_t2x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <6 x i64> %{{.*}} to <6 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <6 x i64> %{{.*}} to <6 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <6 x float> @llvm.atan2.v6f32(<6 x float> [[CONVI]], <6 x float> [[CONV1I]])
// CHECK:    ret <6 x float> [[V5]]
float2x3 test_atan2_uint64_t2x3 (uint64_t2x3 p0, uint64_t2x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <8 x float> @_{{.*}}test_atan2_uint64_t2x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <8 x i64> %{{.*}} to <8 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <8 x i64> %{{.*}} to <8 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <8 x float> @llvm.atan2.v8f32(<8 x float> [[CONVI]], <8 x float> [[CONV1I]])
// CHECK:    ret <8 x float> [[V5]]
float2x4 test_atan2_uint64_t2x4 (uint64_t2x4 p0, uint64_t2x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <3 x float> @_{{.*}}test_atan2_uint64_t3x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <3 x i64> %{{.*}} to <3 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <3 x float> @llvm.atan2.v3f32(<3 x float> [[CONVI]], <3 x float> [[CONV1I]])
// CHECK:    ret <3 x float> [[V5]]
float3x1 test_atan2_uint64_t3x1 (uint64_t3x1 p0, uint64_t3x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <6 x float> @_{{.*}}test_atan2_uint64_t3x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <6 x i64> %{{.*}} to <6 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <6 x i64> %{{.*}} to <6 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <6 x float> @llvm.atan2.v6f32(<6 x float> [[CONVI]], <6 x float> [[CONV1I]])
// CHECK:    ret <6 x float> [[V5]]
float3x2 test_atan2_uint64_t3x2 (uint64_t3x2 p0, uint64_t3x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <9 x float> @_{{.*}}test_atan2_uint64_t3x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <9 x i64> %{{.*}} to <9 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <9 x i64> %{{.*}} to <9 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <9 x float> @llvm.atan2.v9f32(<9 x float> [[CONVI]], <9 x float> [[CONV1I]])
// CHECK:    ret <9 x float> [[V5]]
float3x3 test_atan2_uint64_t3x3 (uint64_t3x3 p0, uint64_t3x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <12 x float> @_{{.*}}test_atan2_uint64_t3x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float3x4 test_atan2_uint64_t3x4 (uint64_t3x4 p0, uint64_t3x4 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <4 x float> @_{{.*}}test_atan2_uint64_t4x1{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <4 x i64> %{{.*}} to <4 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <4 x float> @llvm.atan2.v4f32(<4 x float> [[CONVI]], <4 x float> [[CONV1I]])
// CHECK:    ret <4 x float> [[V5]]
float4x1 test_atan2_uint64_t4x1 (uint64_t4x1 p0, uint64_t4x1 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <8 x float> @_{{.*}}test_atan2_uint64_t4x2{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <8 x i64> %{{.*}} to <8 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <8 x i64> %{{.*}} to <8 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <8 x float> @llvm.atan2.v8f32(<8 x float> [[CONVI]], <8 x float> [[CONV1I]])
// CHECK:    ret <8 x float> [[V5]]
float4x2 test_atan2_uint64_t4x2 (uint64_t4x2 p0, uint64_t4x2 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <12 x float> @_{{.*}}test_atan2_uint64_t4x3{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <12 x i64> %{{.*}} to <12 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <12 x float> @llvm.atan2.v12f32(<12 x float> [[CONVI]], <12 x float> [[CONV1I]])
// CHECK:    ret <12 x float> [[V5]]
float4x3 test_atan2_uint64_t4x3 (uint64_t4x3 p0, uint64_t4x3 p1) {
  return atan2(p0, p1);
}

// CHECK: define [[FNATTRS]] <16 x float> @_{{.*}}test_atan2_uint64_t4x4{{.*}}(
// CHECK:    [[CONVI:%.*]] = uitofp <16 x i64> %{{.*}} to <16 x float>
// CHECK:    [[CONV1I:%.*]] = uitofp <16 x i64> %{{.*}} to <16 x float>
// CHECK:    [[V5:%.*]] = call {{.*}} <16 x float> @llvm.atan2.v16f32(<16 x float> [[CONVI]], <16 x float> [[CONV1I]])
// CHECK:    ret <16 x float> [[V5]]
float4x4 test_atan2_uint64_t4x4 (uint64_t4x4 p0, uint64_t4x4 p1) {
  return atan2(p0, p1);
}
