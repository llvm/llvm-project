// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -hlsl-entry test_mips \
// RUN:   -DTEXTURE=Texture1D -DCOORD_TYPE=float -DINDEX_TYPE=int -o - %s | \
// RUN:   llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SCALAR-COORD \
// RUN:   -DTEXTURE=Texture1D -DCOORD_DIM=1 -DLOAD_DIM=2 -DDXIL_TY=1 -DDIM=1 \
// RUN:   -DCOORD_LLVM=float -DCOORD_CXX=float -DINDEX_LLVM=i32 \
// RUN:   -DINDEX_CXX=int -DOFFSET_LLVM=i32 -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -hlsl-entry test_mips \
// RUN:   -DTEXTURE=Texture1DArray -DCOORD_TYPE=float2 -DINDEX_TYPE=int2 -o - \
// RUN:   %s | llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,COORD2,VEC-COORD -DTEXTURE=Texture1DArray \
// RUN:   -DCOORD_DIM=2 -DLOAD_DIM=3 -DDXIL_TY=6 -DDIM=1 \
// RUN:   -DCOORD_LLVM="<2 x float>" -DCOORD_CXX="float vector[2]" \
// RUN:   -DINDEX_LLVM="<2 x i32>" -DINDEX_CXX="int vector[2]" \
// RUN:   -DOFFSET_LLVM=i32 -DOFFSET_ZERO=0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -hlsl-entry test_mips \
// RUN:   -DTEXTURE=Texture2D -DCOORD_TYPE=float2 -DINDEX_TYPE=int2 -o - %s | \
// RUN:   llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,COORD2,VEC-COORD \
// RUN:   -DTEXTURE=Texture2D -DCOORD_DIM=2 -DLOAD_DIM=3 -DDXIL_TY=2 -DDIM=2 \
// RUN:   -DCOORD_LLVM="<2 x float>" -DCOORD_CXX="float vector[2]" \
// RUN:   -DINDEX_LLVM="<2 x i32>" -DINDEX_CXX="int vector[2]" \
// RUN:   -DOFFSET_LLVM="<2 x i32>" -DOFFSET_ZERO=zeroinitializer
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -finclude-default-header -hlsl-entry test_mips \
// RUN:   -DTEXTURE=Texture2DArray -DCOORD_TYPE=float3 -DINDEX_TYPE=int3 -o - \
// RUN:   %s | llvm-cxxfilt | FileCheck %s \
// RUN:   --check-prefixes=CHECK,COORD3,VEC-COORD -DTEXTURE=Texture2DArray \
// RUN:   -DCOORD_DIM=3 -DLOAD_DIM=4 -DDXIL_TY=7 -DDIM=2 \
// RUN:   -DCOORD_LLVM="<3 x float>" -DCOORD_CXX="float vector[3]" \
// RUN:   -DINDEX_LLVM="<3 x i32>" -DINDEX_CXX="int vector[3]" \
// RUN:   -DOFFSET_LLVM="<2 x i32>" -DOFFSET_ZERO=zeroinitializer

// Parameterized over the texture types in the RUN lines above; adding a texture
// of another dimension only requires new RUN lines.
//
//   TEXTURE            resource type name
//   COORD_TYPE         sample location type
//   INDEX_TYPE         mips slice index type
//   COORD_DIM          sample location components (DIM plus the array slice)
//   COORD_LLVM         sample location type in the IR
//   COORD_CXX          sample location type in the C++ signature
//   INDEX_LLVM         mips slice index type in the IR
//   INDEX_CXX          mips slice index type in the C++ signature
//   OFFSET_LLVM        offset type in the IR
//   OFFSET_ZERO        the all-zero offset as it appears in the IR
//   LOAD_DIM           Load location components (COORD_DIM plus the mip level)
//   DXIL_TY            dx.Texture resource-kind operand
//   DIM                number of resource dimensions (offset, ddx/ddy, LOD
//                      location)
//
// Check prefixes:
//   COORD2             the location is built from two coordinate components
//   COORD3             the location is built from three coordinate components
//   VEC-COORD          the coordinate is a vector, so the location is built
//                      element by element and the coordinate is shuffled back
//                      out of it
//   SCALAR-COORD       1D types, whose coordinate is a single value

TEXTURE<float4> t;

// `mips` caches its own copy of the resource handle, so the initializer has to
// write `__handle` into it as well. Leaving it uninitialized makes
// `t.mips[N][...]` load from a poison handle.
// CHECK: define linkonce_odr hidden void @hlsl::[[TEXTURE]]<float vector[4]>::__createFromImplicitBinding(
// CHECK: %[[NEW_HANDLE:.*]] = call target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]) @llvm.dx.resource.handlefromimplicitbinding
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr {{.*}} %"class.hlsl::[[TEXTURE]]", ptr %[[TMP:.*]], i32 0, i32 0
// CHECK: store target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]) %[[NEW_HANDLE]], ptr %[[HANDLE_GEP]]
// CHECK: %[[HANDLE_GEP2:.*]] = getelementptr {{.*}} %"class.hlsl::[[TEXTURE]]", ptr %[[TMP]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]), ptr %[[HANDLE_GEP2]]
// CHECK: %[[MIPS_GEP:.*]] = getelementptr {{.*}} %"class.hlsl::[[TEXTURE]]", ptr %[[TMP]], i32 0, i32 1
// CHECK: %[[MIPS_HANDLE_GEP:.*]] = getelementptr {{.*}} %"struct.hlsl::[[TEXTURE]]<>::mips_type", ptr %[[MIPS_GEP]], i32 0, i32 0
// CHECK: store target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]) %[[HANDLE]], ptr %[[MIPS_HANDLE_GEP]]

// CHECK: define internal {{.*}} <4 x float> @test_mips([[COORD_CXX]])([[COORD_LLVM]] {{.*}} %loc)
// CHECK: entry:
// CHECK: %[[LOC_ADDR:.*]] = alloca [[COORD_LLVM]]
// CHECK: %[[REF_TMP:.*]] = alloca %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type"
// CHECK: store [[COORD_LLVM]] %loc, ptr %[[LOC_ADDR]]
// CHECK: call void @hlsl::[[TEXTURE]]<float vector[4]>::mips_type::operator[](int) const(ptr {{.*}} %[[REF_TMP]], ptr {{.*}} getelementptr {{.*}} (i8, ptr @t, i32 4), i32 noundef 0)
// CHECK: %[[V0:.*]] = load [[COORD_LLVM]], ptr %[[LOC_ADDR]]
// CHECK: %[[CONV:.*]] = fptosi [[COORD_LLVM]] %[[V0]] to [[INDEX_LLVM]]
// CHECK: %[[CALL:.*]] = call {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::mips_slice_type::operator[]([[INDEX_CXX]]) const(ptr {{.*}} %[[REF_TMP]], [[INDEX_LLVM]] {{.*}} %[[CONV]])
// CHECK: ret <4 x float> %[[CALL]]

[shader("pixel")]
float4 test_mips(COORD_TYPE loc : LOC) : SV_Target {
  return t.mips[0][(INDEX_TYPE)loc];
}

// CHECK: define linkonce_odr hidden void @hlsl::[[TEXTURE]]<float vector[4]>::mips_type::operator[](int) const(ptr  {{.*}} %agg.result, ptr {{.*}} %this, i32 {{.*}} %Level)
// CHECK: entry:
// CHECK: %{{.*}} = alloca ptr
// CHECK: %[[THIS_ADDR:.*]] = alloca ptr
// CHECK: %[[LEVEL_ADDR:.*]] = alloca i32
// CHECK: %[[SLICE:.*]] = alloca %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type"
// CHECK: store ptr %agg.result, ptr %{{.*}}
// CHECK: store ptr %this, ptr %[[THIS_ADDR]]
// CHECK: store i32 %Level, ptr %[[LEVEL_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %[[THIS_ADDR]]
// CHECK: call void @hlsl::[[TEXTURE]]<float vector[4]>::mips_slice_type::mips_slice_type()(ptr {{.*}} %[[SLICE]])
// CHECK: %[[HANDLE_GEP:.*]] = getelementptr {{.*}} %"struct.hlsl::[[TEXTURE]]<>::mips_type", ptr %[[THIS1]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]), ptr %[[HANDLE_GEP]]
// CHECK: %[[HANDLE_GEP2:.*]] = getelementptr {{.*}} %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type", ptr %[[SLICE]], i32 0, i32 0
// CHECK: store target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]) %[[HANDLE]], ptr %[[HANDLE_GEP2]]
// CHECK: %[[L_VAL:.*]] = load i32, ptr %[[LEVEL_ADDR]]
// CHECK: %[[LEVEL_GEP:.*]] = getelementptr {{.*}} %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type", ptr %[[SLICE]], i32 0, i32 1
// CHECK: store i32 %[[L_VAL]], ptr %[[LEVEL_GEP]]
// CHECK: call void @hlsl::[[TEXTURE]]<float vector[4]>::mips_slice_type::mips_slice_type(hlsl::[[TEXTURE]]<float vector[4]>::mips_slice_type const&)(ptr noundef nonnull align 4 dereferenceable(8) %agg.result, ptr noundef nonnull align 4 dereferenceable(8) %[[SLICE]])

// CHECK: define linkonce_odr hidden {{.*}} <4 x float> @hlsl::[[TEXTURE]]<float vector[4]>::mips_slice_type::operator[]([[INDEX_CXX]]) const(ptr {{.*}} %[[THIS:.*]], [[INDEX_LLVM]] noundef %[[COORD:.*]])
// CHECK: entry:
// CHECK: %[[COORD_ADDR:.*]] = alloca [[INDEX_LLVM]]
// VEC-COORD: %[[VEC_TMP:.*]] = alloca [[INDEX_LLVM]]
// CHECK: store [[INDEX_LLVM]] %[[COORD]], ptr %[[COORD_ADDR]]
// CHECK: %[[THIS1:.*]] = load ptr, ptr %{{.*}}
// VEC-COORD: %[[COORD_PARAM:.*]] = load [[INDEX_LLVM]], ptr %[[COORD_ADDR]]
// VEC-COORD: store [[INDEX_LLVM]] %[[COORD_PARAM]], ptr %[[VEC_TMP]]
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr {{.*}} %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type", ptr %[[THIS1]], i32 0, i32 0
// CHECK: %[[HANDLE:.*]] = load target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]), ptr %[[HANDLE_PTR]]
// A 1D coordinate is a single value, so it goes straight into the location.
// SCALAR-COORD: %[[COORD_VAL:.*]] = load i32, ptr %[[COORD_ADDR]]
// SCALAR-COORD: %[[VECINIT3:.*]] = insertelement <[[LOAD_DIM]] x i32> poison, i32 %[[COORD_VAL]], i32 0

// VEC-COORD: %[[COORD_VAL:.*]] = load [[INDEX_LLVM]], ptr %[[VEC_TMP]]
// VEC-COORD: %[[VECEXT:.*]] = extractelement [[INDEX_LLVM]] %[[COORD_VAL]], i32 0
// VEC-COORD: %[[VECINIT:.*]] = insertelement <[[LOAD_DIM]] x i32> poison, i32 %[[VECEXT]], i32 0
// VEC-COORD: %[[COORD_VAL2:.*]] = load [[INDEX_LLVM]], ptr %[[VEC_TMP]]
// VEC-COORD: %[[VECEXT2:.*]] = extractelement [[INDEX_LLVM]] %[[COORD_VAL2]], i32 1
// VEC-COORD: %[[VECINIT3:.*]] = insertelement <[[LOAD_DIM]] x i32> %[[VECINIT]], i32 %[[VECEXT2]], i32 1
// COORD3: %[[COORD_VAL3:.*]] = load <3 x i32>, ptr %[[VEC_TMP]]
// COORD3: %[[VECEXT3:.*]] = extractelement <3 x i32> %[[COORD_VAL3]], i32 2
// COORD3: %[[VECINIT3B:.*]] = insertelement <4 x i32> %[[VECINIT3]], i32 %[[VECEXT3]], i32 2
// CHECK: %[[LEVEL_PTR:.*]] = getelementptr {{.*}} %"struct.hlsl::[[TEXTURE]]<>::mips_slice_type", ptr %[[THIS1]], i32 0, i32 1
// CHECK: %[[LEVEL_VAL:.*]] = load i32, ptr %[[LEVEL_PTR]]
// SCALAR-COORD: %[[VECINITL:.*]] = insertelement <2 x i32> %[[VECINIT3]], i32 %[[LEVEL_VAL]], i32 1
// COORD2: %[[VECINITL:.*]] = insertelement <3 x i32> %[[VECINIT3]], i32 %[[LEVEL_VAL]], i32 2
// COORD3: %[[VECINITL:.*]] = insertelement <4 x i32> %[[VECINIT3B]], i32 %[[LEVEL_VAL]], i32 3
// SCALAR-COORD: %[[COORD_X:.*]] = extractelement <[[LOAD_DIM]] x i32> %[[VECINITL]], i64 0
// VEC-COORD: %[[COORD_X:.*]] = shufflevector <[[LOAD_DIM]] x i32> %[[VECINITL]], <[[LOAD_DIM]] x i32> poison, [[INDEX_LLVM]] {{.*}}
// CHECK: %[[LOD:.*]] = extractelement <[[LOAD_DIM]] x i32> %[[VECINITL]], i64 [[COORD_DIM]]
// CHECK: %[[RES:.*]] = call {{.*}} <4 x float> @llvm.dx.resource.load.level.v4f32.tdx.Texture_v4f32_0_0_0_[[DXIL_TY]]t{{.*}}(target("dx.Texture", <4 x float>, 0, 0, 0, [[DXIL_TY]]) %[[HANDLE]], [[INDEX_LLVM]] %[[COORD_X]], i32 %[[LOD]], [[OFFSET_LLVM]] [[OFFSET_ZERO]])
// CHECK: ret <4 x float> %[[RES]]
