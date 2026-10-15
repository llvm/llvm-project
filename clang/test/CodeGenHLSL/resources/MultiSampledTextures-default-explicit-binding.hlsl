// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DMS -DLOCATION_TYPE=int2 -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL -DTEXTURE=Texture2DMS -DCOORD_DIM=2 -DKIND=3
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -finclude-default-header -DTEXTURE=Texture2DMS -DLOCATION_TYPE=int2 -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV -DTEXTURE=Texture2DMS -DCOORD_DIM=2 -DARRAYED=0

// A multisampled texture with an explicit register/space binding is initialized
// from that binding through __createFromBinding, identically to a non-MS
// texture; only the resource handle type differs (dx.MSTexture / a multisampled
// spirv.Image). Unlike non-multisampled textures, the element type must be
// stated explicitly (Texture2DMS and Texture2DMS<> are both errors), so the
// default-template / shorthand forms are not exercised here.

TEXTURE<float4> explicit_binding : register(t1, space2);
TEXTURE<float4> implicit_template : register(t0, space1);

// DXIL: %"class.hlsl::[[TEXTURE]]" = type { target("dx.MSTexture", <4 x float>, 0, 0, 0, [[KIND]]) }
// SPIRV: %"class.hlsl::[[TEXTURE]]" = type { target("spirv.Image", float, 1, 2, [[ARRAYED]], 1, 1, 0) }

// CHECK: @{{.*}}explicit_binding = internal global %"class.hlsl::[[TEXTURE]]" poison, align {{[0-9]+}}
// CHECK: @{{.*}}implicit_template = internal global %"class.hlsl::[[TEXTURE]]" poison, align {{[0-9]+}}

// Each texture is initialized from its explicit register/space binding:
//   explicit_binding  -> register(t1, space2)  =>  registerNo 1, spaceNo 2
//   implicit_template -> register(t0, space1)  =>  registerNo 0, spaceNo 1
// CHECK: call {{.*}} @hlsl::[[TEXTURE]]<float vector[4]{{(, [0-9]+)?}}>::__createFromBinding{{.*}}(ptr {{.*}}@{{.*}}explicit_binding, i32 noundef 1, i32 noundef 2, i32 noundef 1, i32 noundef 0, ptr noundef @{{.*}})
// CHECK: call {{.*}} @hlsl::[[TEXTURE]]<float vector[4]{{(, [0-9]+)?}}>::__createFromBinding{{.*}}(ptr {{.*}}@{{.*}}implicit_template, i32 noundef 0, i32 noundef 1, i32 noundef 1, i32 noundef 0, ptr noundef @{{.*}})

float4 main(LOCATION_TYPE loc : LOC, int sampleIndex : SI) : SV_Target {
  // A multisampled texture is read through a sample-indexed Load, not Sample.
  // DXIL: call {{.*}} <4 x float> @llvm.dx.resource.load.ms.v4f32.{{.*}}(target("dx.MSTexture", <4 x float>, 0, 0, 0, [[KIND]]) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
  // SPIRV: call {{.*}} <4 x float> @llvm.spv.resource.load.ms.v4f32.{{.*}}(target("spirv.Image", float, 1, 2, [[ARRAYED]], 1, 1, 0) %{{.*}}, <[[COORD_DIM]] x i32> %{{.*}}, i32 %{{.*}}, <2 x i32> zeroinitializer)
  return implicit_template.Load(loc, sampleIndex);
}
