// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DMS -DLOCATION_TYPE=int2 -o - %s | FileCheck %s --check-prefix=DXIL -DCOORD_DIM=2 -DKIND=3
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DMS -DLOCATION_TYPE=int2 -o - %s | FileCheck %s --check-prefix=SPIRV -DCOORD_DIM=2 -DARRAYED=0

// Load on a multisampled texture takes a separate sample index (rather than a
// packed mip level) and lowers to the MS-specific resource.load.ms intrinsic on
// a dx.MSTexture (DXIL) / multisampled spirv.Image (SPIR-V) handle. The correct
// per-element-type intrinsic overload is selected, and signed integer element
// types map to a SignedImage / IsSigned=1 handle. The resource type layout
// itself is covered by MultiSampledTextures-default.hlsl.

TEXTURE<float4> T;
TEXTURE<float4, 4> TMS4;
TEXTURE<float> Tf;
TEXTURE<int> Ti;
TEXTURE<int4> Ti4;

// CHECK-LABEL: define {{.*}} <4 x float> @{{.*}}test_load
// DXIL: call {{.*}} <4 x float> @llvm.dx.resource.load.ms.v4f32.{{.*}}(target("dx.MSTexture", <4 x float>, 0, 0, 0, [[KIND]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: call {{.*}} <4 x float> @llvm.spv.resource.load.ms.v4f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 1, 1, 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
float4 test_load(LOCATION_TYPE loc, int sampleIndex) {
  return T.Load(loc, sampleIndex);
}

// The offset overload threads the constant offset through as the final operand.
// CHECK-LABEL: define {{.*}} <4 x float> @{{.*}}test_load_offset
// DXIL: call {{.*}} <4 x float> @llvm.dx.resource.load.ms.v4f32.{{.*}}(target("dx.MSTexture", <4 x float>, 0, 0, 0, [[KIND]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
// SPIRV: call {{.*}} <4 x float> @llvm.spv.resource.load.ms.v4f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 1, 1, 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> %{{.*}})
float4 test_load_offset(LOCATION_TYPE loc, int sampleIndex) {
  return T.Load(loc, sampleIndex, int2(1, 1));
}

// An explicit compile-time sample count changes only the resource handle type
// (the dx.MSTexture sample-count operand becomes 4); the resource.load.ms call
// is otherwise identical to the default.
// CHECK-LABEL: define {{.*}} <4 x float> @{{.*}}test_explicit_count
// DXIL: call {{.*}} <4 x float> @llvm.dx.resource.load.ms.v4f32.{{.*}}(target("dx.MSTexture", <4 x float>, 0, 4, 0, [[KIND]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: call {{.*}} <4 x float> @llvm.spv.resource.load.ms.v4f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 1, 1, 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
float4 test_explicit_count(LOCATION_TYPE loc, int sampleIndex) {
  return TMS4.Load(loc, sampleIndex);
}

// A scalar float element selects the f32 overload.
// CHECK-LABEL: define {{.*}} float @{{.*}}test_load_float
// DXIL: call {{.*}} float @llvm.dx.resource.load.ms.f32.{{.*}}(target("dx.MSTexture", float, 0, 0, 0, [[KIND]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: call {{.*}} float @llvm.spv.resource.load.ms.f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 1, 1, 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
float test_load_float(LOCATION_TYPE loc, int sampleIndex) {
  return Tf.Load(loc, sampleIndex);
}

// A signed integer element maps to IsSigned=1 (DXIL) / spirv.SignedImage.
// CHECK-LABEL: define {{.*}} i32 @{{.*}}test_load_int
// DXIL: call i32 @llvm.dx.resource.load.ms.i32.{{.*}}(target("dx.MSTexture", i32, 0, 0, 1, [[KIND]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: call i32 @llvm.spv.resource.load.ms.i32.{{.*}}(target("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 1, 1, 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
int test_load_int(LOCATION_TYPE loc, int sampleIndex) {
  return Ti.Load(loc, sampleIndex);
}

// A signed integer vector element: v4i32 overload on a SignedImage handle.
// CHECK-LABEL: define {{.*}} <4 x i32> @{{.*}}test_load_int4
// DXIL: call <4 x i32> @llvm.dx.resource.load.ms.v4i32.{{.*}}(target("dx.MSTexture", <4 x i32>, 0, 0, 1, [[KIND]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
// SPIRV: call <4 x i32> @llvm.spv.resource.load.ms.v4i32.{{.*}}(target("spirv.SignedImage", i32, 1, 2, [[ARRAYED]], 1, 1, 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
int4 test_load_int4(LOCATION_TYPE loc, int sampleIndex) {
  return Ti4.Load(loc, sampleIndex);
}
